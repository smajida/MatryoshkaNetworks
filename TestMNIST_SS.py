import os
import json
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import sys
sys.setrecursionlimit(100000)

import theano
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.ops import log_mean_exp, binarize_data
from lib.costs import log_prob_bernoulli
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data, OneHot
from load import load_udm_ss

#
# Phil's business
#
from MatryoshkaModules import BasicConvModule, GenConvResModule, \
                              GenFCModule, InfConvMergeModule, \
                              InfFCModule, BasicConvResModule, \
                              MlpFCModule
from MatryoshkaNetworks import InfGenModelSS, SimpleInfMLP

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
sup_count = 100
desc = "test_ss_{}_labels".format(sup_count)
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load binarized MNIST dataset
data_path = "{}/data/mnist.pkl.gz".format(EXP_DIR)
data_dict = load_udm_ss(data_path, sup_count)
Xtr_su, Ytr_su = data_dict['Xtr_su'], data_dict['Ytr_su']
Xtr_un, Ytr_un = data_dict['Xtr_un'], data_dict['Ytr_un']
Xva, Yva = data_dict['Xva'], data_dict['Yva']
Xte, Yte = data_dict['Xte'], data_dict['Yte']

Xtr_un = np.concatenate([Xtr_un, Xtr_su], axis=0)
Ytr_un = np.concatenate([Ytr_un, Ytr_su], axis=0)

# convert labels to one-hot representation
nyc = 10
Ytr_su = floatX( OneHot(Ytr_su, n=nyc, negative_class=0.) )
Ytr_un = floatX( OneHot(Ytr_un, n=nyc, negative_class=0.) )
Yva = floatX( OneHot(Yva, n=nyc, negative_class=0.) )
Yte = floatX( OneHot(Yte, n=nyc, negative_class=0.) )

sup_count = Xtr_su.shape[0]   # number of labeled examples

set_seed(1)       # seed for shared rngs
nc = 1            # # of channels in image
nbatch = 50       # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 32          # # of dim for Z0 (latents at top of generator)
nz1 = 16          # # of dim for Z1 (latents in conv layers of generator)
nza = 32          # dimension of "auxiliary" latent variables
ngf = 32          # base # of filters for conv layers in generative stuff
ngfc = 128        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
niter = 200       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_bn = True     # whether to use batch normalization throughout the model
act_func = 'relu' # activation func to use where they can be selected

def class_accuracy(Y_model, Y_true, return_raw_count=False):
    """
    Check the accuracy of predictions in Y_model, given the ground truth in
    Y_true. Make predictions by row-wise argmax, and assume Y_true is one-hot.
    """
    yt = np.argmax(Y_true, axis=1)
    ym = np.argmax(Y_model, axis=1)
    hits = float( np.sum((yt == ym)) )
    if return_raw_count:
        res = hits
    else:
        res = hits / Y_true.shape[0]
    return res

def train_transform(X):
    # transform vectorized observations into convnet inputs
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3))

def draw_transform(X):
    # transform vectorized observations into drawable greyscale images
    X = X * 255.0
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1))

def rand_gen(size, noise_type='normal'):
    if noise_type == 'normal':
        r_vals = floatX(np_rng.normal(size=size))
    elif noise_type == 'uniform':
        r_vals = floatX(np_rng.uniform(size=size, low=-1.0, high=1.0))
    else:
        assert False, "unrecognized noise type!"
    return r_vals

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

#########################################
# Setup the top-down processing modules #
# -- these do generation                #
#########################################

# FC -> (7, 7)
td_module_1 = \
GenFCModule(
    rand_dim=nz0+nyc,
    out_shape=(ngf*4, 7, 7),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=use_bn,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2 = \
GenConvResModule(
    in_chans=(ngf*4)+nyc,
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=1,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (14, 14)
td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4)+nyc,
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (28, 28)
td_module_4 = \
GenConvResModule(
    in_chans=(ngf*2)+nyc,
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_4'
) # output is (batch, ngf*1, 28, 28)

# (28, 28) -> (28, 28)
td_module_5 = \
BasicConvModule(
    in_chans=(ngf*1),
    out_chans=nc,
    filt_shape=(3,3),
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_5'
) # output is (batch, c, 28, 28)

# modules must be listed in "evaluation order"
td_modules = [td_module_1, td_module_2, td_module_3, td_module_4, td_module_5]



###############################################################################
# Setup the three inference models that sit on top of the BU network. These   #
# models provide distributions over auxiliary latent variable "a", the class  #
# indicator variable "y", and the initial primary latent variables "z0".      #
###############################################################################

#
# modules and MLP for conditional q(a | x)
#
# this receives deterministic input from the top of the BU network.
#
# this provides a distribution/sample for "auxiliary" latent variable "a".
#
q_aIx_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*7*7), # input from BU net only
    fc_chans=ngfc,
    rand_chans=nza,       # output is mean and logvar for "a"
    use_fc=True,
    unif_drop=0.0,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='q_aIx_module_1'
)
q_aIx_modules = [ q_aIx_module_1 ]
q_aIx_model = SimpleInfMLP(modules=q_aIx_modules)

#
# module for conditional q(y | a, x)
#
# this receives deterministic input from the top of the BU network, and a
# sample of the auxiliary latent variable "a".
#
# this provides an output of dimension "nyc" (the number of classes), which
# will be softmaxed by the model which handles the overall generative system.
#
q_yIax_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*7*7)+nza, # input from BU net and "a"
    fc_chans=ngfc,
    rand_chans=nyc,           # output is (unnormalized) class distribution
    use_fc=True,
    unif_drop=0.0,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='q_yIax_module_1'
)
q_yIax_modules = [ q_yIax_module_1 ]
q_yIax_model = SimpleInfMLP(modules=q_yIax_modules)

#
# module for conditional q(z | y, x)
#
# this receives deterministic input from the top of the BU network, and a
# class indicator vector. we'll marginalize over possible values of the class
# indicator during inference.
#
# this provides a sample of the initial "primary" latent variable "z0", which
# goes into the top of the TD network (in addition to a class indicator).
#
q_z0Iyx_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*7*7)+nyc, # input from BU net and class indicator
    fc_chans=ngfc,
    rand_chans=nz0,           # output is mean and logvar for "z0"           
    use_fc=True,
    unif_drop=0.0,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='q_z0Iyx_module_1'
)
q_z0Iyx_modules = [ q_z0Iyx_module_1 ]
q_z0Iyx_model = SimpleInfMLP(modules=q_z0Iyx_modules)

#############################################################################
# Setup the modules for the bottom-up convolutional network which extracts  #
# features for use by the top-most modules in the inference network.        #
#############################################################################

# (7, 7) -> (7, 7)
bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.0,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2'
) # output is (batch, ngf*4, 7, 7)

# (14, 14) -> (7, 7)
bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.0,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 7, 7)

# (28, 28) -> (14, 14)
bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.0,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_4'
) # output is (batch, ngf*2, 14, 14)

# (28, 28) -> (28, 28)
bu_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    chan_drop=0.0,
    unif_drop=0.0,
    apply_bn=False,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_5'
) # output is (batch, ngf*1, 28, 28)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_5, bu_module_4, bu_module_3, bu_module_2]

#########################################
# Setup the information merging modules #
#########################################

im_module_2 = \
InfConvMergeModule(
    td_chans=(ngf*4)+nyc, # this gets to see the indicators
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*4),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=False,
    act_func=act_func,
    mod_name='im_mod_2'
)

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4)+nyc, # this gets to see the indicators
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=False,
    act_func=act_func,
    mod_name='im_mod_3'
)

im_module_4 = \
InfConvMergeModule(
    td_chans=(ngf*2)+nyc, # this gets to see the indicators
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=False,
    act_func=act_func,
    mod_name='im_mod_4'
)

im_modules = [im_module_2, im_module_3, im_module_4]

#
# Setup a description for where to get conditional distributions from. When
# there's no info here for a particular top-down module.
#
merge_info = {
    'td_mod_2': {'bu_module': 'bu_mod_2', 'im_module': 'im_mod_2'},
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
    'td_mod_4': {'bu_module': 'bu_mod_4', 'im_module': 'im_mod_4'},
}

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModelSS(
    nyc=nyc,                     # number of classes (i.e. indicator dim)
    nbatch=nbatch,               # _fixed_ batch size for "marginalized" TD/BU passes
    q_aIx_model=q_aIx_model,     # model for q(a | x)
    q_yIax_model=q_yIax_model,   # model for q(y | a, x)
    q_z0Iyx_model=q_z0Iyx_model, # model for q(z0 | y, x)
    bu_modules=bu_modules,       # modules for BU (inference) conv net
    td_modules=td_modules,       # modules for TD (generator) conv net
    im_modules=im_modules,       # modules for q(zi | y, x) within TD conv net
    merge_info=merge_info,       # piping for input/output of TD/IM/BU modules
    output_transform=sigmoid     # transform to apply to generator output
)

# quick save/load test
inf_gen_model.dump_params(f_name=inf_gen_param_file)
inf_gen_model.load_params(f_name=inf_gen_param_file)


####################################
# Setup the optimization objective #
####################################
lam_un_vae = sharedX(floatX(np.asarray([1.0])))
lam_su_vae = sharedX(floatX(np.asarray([1.0])))
lam_su_cls = sharedX(floatX(np.asarray([1.0])))
all_params = inf_gen_model.params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

#
# The cost for a labeled point (x, y) is:
#   NLL_bound(x, y) - lam_su_cls*NLL(q(y|x))
#
# The cost for an unlabeled point x is:
#   NLL_bound(x)
#
# The overall cost is given by:
#  
#

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs for unsupervised loss
Xc = T.tensor4()  # symbolic var for inputs for supervised loss
Yc = T.matrix()   # symbolic var for one-hot labels for supervised loss
Z0 = T.matrix()   # symbolic var for top-most latent variables for generator

# quick test of the marginalized BU/TD inference process
im_res_dict = inf_gen_model.apply_im_unlabeled_1(Xg)

obs_nlls = im_res_dict['obs_nlls']
obs_klds = im_res_dict['obs_klds']
log_p_xIz = im_res_dict['log_p_xIz']
kld_z = im_res_dict['kld_z']
kld_a = im_res_dict['kld_a']
kld_y = im_res_dict['kld_y']
ent_y = im_res_dict['ent_y']

test_func = theano.function([Xg], [obs_nlls, obs_klds])

obs_nlls, obs_klds = test_func(train_transform(Xtr[0:nbatch,:]))

# quick test of the other marginalized BU/TD inference process
im_res_dict = inf_gen_model.apply_im_unlabeled_2(Xg)

obs_nlls = im_res_dict['obs_nlls']
obs_klds = im_res_dict['obs_klds']
log_p_xIz = im_res_dict['log_p_xIz']
kld_z = im_res_dict['kld_z']
kld_a = im_res_dict['kld_a']
kld_y = im_res_dict['kld_y']
ent_y = im_res_dict['ent_y']

test_func = theano.function([Xg], [obs_nlls, obs_klds])

obs_nlls, obs_klds = test_func(train_transform(Xtr[0:nbatch,:]))

# ##########################################################
# # CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
# ##########################################################
# # run an inference and reconstruction pass through the generative stuff
# im_res_dict = inf_gen_model.apply_im(Xg)
# Xg_recon = im_res_dict['td_output']
# vae_kld_dict = im_res_dict['kld_dict']
# vae_log_p_x = T.sum(log_prob_bernoulli( \
#                     T.flatten(Xg,2), T.flatten(Xg_recon,2),
#                     do_sum=False), axis=1)

# # compute reconstruction error part of free-energy
# vae_obs_nlls = -1.0 * vae_log_p_x
# vae_nll_cost = T.mean(vae_obs_nlls)

# # compute per-layer KL-divergence part of cost
# vae_kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) \
#                   for mod_name, mod_kld in vae_kld_dict.items()]
# vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) \
#                   for mod_name, mod_kld in vae_kld_tuples])
# vae_layer_names = [mod_name for mod_name, mod_kld in vae_kld_tuples]
# # compute total per-observation KL-divergence part of cost
# vae_obs_klds = sum([mod_kld for mod_name, mod_kld in vae_kld_tuples])
# vae_kld_cost = T.mean(vae_obs_klds)

# # combined cost for free-energy stuff
# vae_cost = vae_nll_cost + vae_kld_cost

# ##########################################################
# # CONSTRUCT COST VARIABLES FOR THE CLS PART OF OBJECTIVE #
# ##########################################################
# # run an inference and reconstruction pass through the generative stuff
# im_res_dict = inf_gen_model.apply_im(Xc)
# Xc_recon = im_res_dict['td_output']
# cls_kld_dict = im_res_dict['kld_dict']
# cls_z_dict = im_res_dict['z_dict']
# cls_z_top = cls_z_dict['td_mod_1']

# # apply classifier to the top-most set of latent variables, maybe good idea?
# Yc_recon = T.nnet.softmax( class_model.apply(cls_z_top[:,:nz0_c]) )

# # compute reconstruction error part of free-energy
# cls_log_p_x = T.sum(log_prob_bernoulli( \
#                     T.flatten(Xc,2), T.flatten(Xc_recon,2),
#                     do_sum=False), axis=1)
# cls_obs_nlls = -1.0 * cls_log_p_x
# cls_nll_cost = T.mean(cls_obs_nlls)

# # compute per-layer KL-divergence part of cost
# cls_kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) \
#                   for mod_name, mod_kld in cls_kld_dict.items()]
# cls_layer_klds = T.as_tensor_variable([T.mean(mod_kld) \
#                   for mod_name, mod_kld in cls_kld_tuples])
# cls_layer_names = [mod_name for mod_name, mod_kld in cls_kld_tuples]
# # compute total per-observation KL-divergence part of cost
# cls_obs_klds = sum([mod_kld for mod_name, mod_kld in cls_kld_tuples])
# cls_kld_cost = T.mean(cls_obs_klds)

# # combined cost for generator stuff
# cls_cls_cost = lam_cls_cls[0] * T.mean(T.nnet.categorical_crossentropy(Yc_recon, Yc))
# cls_cost = cls_nll_cost + cls_kld_cost + cls_cls_cost + \
#            (T.mean(Yc**2.0) * T.sum(cls_z_dict['td_mod_1']**2.0))

# #################################################################
# # COMBINE VAE AND CLS OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
# #################################################################
# # parameter regularization part of cost
# reg_cost = 2e-5 * sum([T.sum(p**2.0) for p in all_params])
# # full joint cost -- a weighted combination of free-energy and classification
# full_cost = (lam_vae[0] * vae_cost) + (lam_cls[0] * cls_cost) + reg_cost

# # run an un-grounded pass through generative stuff for sampling from model
# td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
# Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

# # stuff for performing updates
# lrt = sharedX(0.0002)
# b1t = sharedX(0.8)
# gen_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)
# inf_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)
# cls_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)


# # build training cost and update functions
# t = time()
# print("Computing gradients...")
# gen_updates, gen_grads = gen_updater(gen_params, full_cost, return_grads=True)
# inf_updates, inf_grads = inf_updater(inf_params, full_cost, return_grads=True)
# cls_updates, cls_grads = inf_updater(cls_params, full_cost, return_grads=True)
# all_updates = gen_updates + inf_updates + cls_updates
# gen_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in gen_grads]))
# inf_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in inf_grads]))
# print("Compiling sampling and reconstruction functions...")
# recon_func = theano.function([Xg], Xg_recon)
# sample_func = theano.function([Z0], Xd_model)
# test_recons = recon_func(train_transform(Xtr_un[0:100,:]))
# print("Compiling training functions...")
# # collect costs for generator parameters
# train_costs = [full_cost, vae_cost, cls_cost, vae_nll_cost, vae_kld_cost,
#                cls_nll_cost, cls_kld_cost, cls_cls_cost, vae_layer_klds]

# tc_idx = range(0, len(train_costs))
# tc_names = ['full_cost', 'vae_cost', 'cls_cost', 'vae_nll_cost',
#               'vae_kld_cost', 'cls_nll_cost', 'cls_kld_cost',
#               'cls_cls_cost', 'vae_layer_klds']
# # compile functions for computing generator costs and updates
# cost_func = theano.function([Xg, Xc, Yc], train_costs)
# pred_func = theano.function([Xc], Yc_recon)
# train_func = theano.function([Xg, Xc, Yc], train_costs, updates=all_updates)
# print "{0:.2f} seconds to compile theano functions".format(time()-t)

# # make file for recording test progress
# log_name = "{}/RESULTS.txt".format(result_dir)
# out_file = open(log_name, 'wb')

# print("EXPERIMENT: {}".format(desc.upper()))

# n_check = 0
# n_updates = 0
# t = time()
# ntrain = Xtr_un.shape[0]
# batch_idx_un = np.arange(ntrain)
# batch_idx_un = np.concatenate([batch_idx_un[:,np.newaxis],batch_idx_un[:,np.newaxis]], axis=1)
# for epoch in range(1, niter+niter_decay+1):
#     Xtr_un = shuffle(Xtr_un)
#     # set relative weights of objectives
#     lam_vae.set_value(floatX(np.asarray([1.0])))
#     lam_cls.set_value(floatX(np.asarray([0.1])))
#     lam_cls_cls.set_value(floatX(np.asarray([10.0])))
#     # initialize cost arrays
#     epoch_costs = [0. for i in range(8)]
#     val_epoch_costs = [0. for i in range(8)]
#     epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
#     batch_count = 0.
#     val_batch_count = 0.
#     train_acc = 0.
#     val_acc = 0.
#     for bidx in tqdm(iter_data(batch_idx_un, size=nbatch), total=ntrain/nbatch):
#         # grab a validation batch, if required
#         if val_batch_count < 50:
#             start_idx = int(val_batch_count)*nbatch
#             vmb_x = Xva[start_idx:(start_idx+nbatch),:].copy()
#             vmb_y = Yva[start_idx:(start_idx+nbatch),:].copy()
#         else:
#             vmb_x = Xva[0:nbatch,:].copy()
#             vmb_y = Yva[0:nbatch,:].copy()
#         # transform training batch to "image format"
#         bidx = bidx[:,0].ravel()
#         imb_x = Xtr_un[bidx,:].copy()
#         cmb_x = Xtr_su[0:100,:].copy()
#         cmb_y = Ytr_su[0:100,:].copy()
#         imb_x_img = train_transform(imb_x)
#         cmb_x_img = train_transform(cmb_x)
#         vmb_x_img = train_transform(vmb_x)
#         # train vae on training batch
#         result = train_func(imb_x_img, cmb_x_img, cmb_y)
#         epoch_costs = [(v1 + v2) for v1, v2 in zip(result[:8], epoch_costs)]
#         epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(result[8], epoch_layer_klds)]
#         y_pred = pred_func(cmb_x_img)
#         train_acc += class_accuracy(y_pred, cmb_y, return_raw_count=False)
#         batch_count += 1
#         # evaluate vae on validation batch
#         if val_batch_count < 25:
#             val_result = train_func(vmb_x_img, vmb_x_img, vmb_y)
#             val_epoch_costs = [(v1 + v2) for v1, v2 in zip(val_result[:8], val_epoch_costs)]
#             y_pred = pred_func(vmb_x_img)
#             val_acc += class_accuracy(y_pred, vmb_y, return_raw_count=False)
#             val_batch_count += 1
#     if (epoch == 20) or (epoch == 50) or (epoch == 100) or (epoch == 200):
#         # cut learning rate in half
#         lr = lrt.get_value(borrow=False)
#         lr = lr / 2.0
#         lrt.set_value(floatX(lr))
#         b1 = b1t.get_value(borrow=False)
#         b1 = b1 + ((0.95 - b1) / 2.0)
#         b1t.set_value(floatX(b1))
#     if epoch > niter:
#         # linearly decay learning rate
#         lr = lrt.get_value(borrow=False)
#         remaining_epochs = (niter + niter_decay + 1) - epoch
#         lrt.set_value(floatX(lr - (lr / remaining_epochs)))
#     ###################
#     # SAVE PARAMETERS #
#     ###################
#     inf_gen_model.dump_params(inf_gen_param_file)
#     ##################################
#     # QUANTITATIVE DIAGNOSTICS STUFF #
#     ##################################
#     epoch_costs = [(c / batch_count) for c in epoch_costs]
#     epoch_layer_klds = [(c / batch_count) for c in epoch_layer_klds]
#     train_acc = train_acc / batch_count
#     val_epoch_costs = [(c / val_batch_count) for c in val_epoch_costs]
#     val_acc = val_acc / val_batch_count
#     str1 = "Epoch {}:".format(epoch)
#     tc_strs = ["{0:s}: {1:.2f},".format(c_name, epoch_costs[c_idx]) \
#                  for (c_idx, c_name) in zip(tc_idx[:8], tc_names[:8])]
#     str2 = " ".join(tc_strs)
#     kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
#     str3 = "    module kld -- {}".format(" ".join(kld_strs))
#     str4 = "    train_acc: {0:.4f}, val_acc: {1:.4f}, val_cls_cls_cost: {2:.4f}".format( \
#             train_acc, val_acc, val_epoch_costs[7])
#     joint_str = "\n".join([str1, str2, str3, str4])
#     print(joint_str)
#     out_file.write(joint_str+"\n")
#     out_file.flush()
#     #################################
#     # QUALITATIVE DIAGNOSTICS STUFF #
#     #################################
#     # generate some samples from the model prior
#     sample_z0mb = np.repeat(rand_gen(size=(20, nz0)), 20, axis=0)
#     samples = np.asarray(sample_func(sample_z0mb))
#     grayscale_grid_vis(draw_transform(samples), (20, 20), "{}/gen_{}.png".format(result_dir, epoch))
#     # test reconstruction performance (inference + generation)
#     tr_rb = Xtr_un[0:100,:]
#     va_rb = Xva[0:100,:]
#     # get the model reconstructions
#     tr_rb = train_transform(tr_rb)
#     va_rb = train_transform(va_rb)
#     tr_recons = recon_func(tr_rb)
#     va_recons = recon_func(va_rb)
#     # stripe data for nice display (each reconstruction next to its target)
#     tr_vis_batch = np.zeros((200, nc, npx, npx))
#     va_vis_batch = np.zeros((200, nc, npx, npx))
#     for rec_pair in range(100):
#         idx_in = 2*rec_pair
#         idx_out = 2*rec_pair + 1
#         tr_vis_batch[idx_in,:,:,:] = tr_rb[rec_pair,:,:,:]
#         tr_vis_batch[idx_out,:,:,:] = tr_recons[rec_pair,:,:,:]
#         va_vis_batch[idx_in,:,:,:] = va_rb[rec_pair,:,:,:]
#         va_vis_batch[idx_out,:,:,:] = va_recons[rec_pair,:,:,:]
#     # draw images...
#     grayscale_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(result_dir, epoch))
#     grayscale_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(result_dir, epoch))







##############
# EYE BUFFER #
##############
