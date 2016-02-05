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

# seed for shared rngs
set_seed(1)

# setup paths for dumping diagnostic info
sup_count = 100
desc = "test_ss_{}_labels_relu_bn_noise_010".format(sup_count)
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
drop_rate = 0.0   # dropout rate in BU network
noise_lvl = 0.1   # noise level in BU network
act_func = 'relu' # activation func to use where they can be selected

def shared_shuffle(x1, x2):
    """
    Shuffle matrices x1 and x2 along axis 0 using the same permutation.
    """
    idx = np.arange(x1.shape[0])
    npr.shuffle(idx)
    x1 = x1[idx,:]
    x2 = x2[idx,:]
    return x1, x2

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
    unif_drop=drop_rate,
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
    unif_drop=drop_rate,
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
    unif_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    unif_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    chan_drop=drop_rate,
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
    use_bernoulli=True,          # flag for using bernoulli/gaussian output
    output_transform=sigmoid     # transform to apply to generator output
)

# quick save/load test
inf_gen_model.dump_params(f_name=inf_gen_param_file)
inf_gen_model.load_params(f_name=inf_gen_param_file)


####################################
# Setup the optimization objective #
####################################
lam_un = sharedX(floatX(np.asarray([1.0])))     # weighting param for unsupervised free-energy
lam_su = sharedX(floatX(np.asarray([0.5])))     # weighting param for total labeled cost
lam_su_cls = sharedX(floatX(np.asarray([0.1])))  # weighting param for classification part of labeled cost
lam_obs_ent_y = sharedX(floatX(np.asarray([0.0])))     # weighting param for observation-wise entropy
lam_batch_ent_y = sharedX(floatX(np.asarray([-5.0])))  # weighting param for batch-wise entropy
bu_noise = sharedX(floatX(np.asarray([noise_lvl])))
all_params = inf_gen_model.params


########################################################
# GATHER SYMBOLIC OUTPUTS REQUIRED FOR COMPUTING COSTS #
########################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs for unsupervised loss
Xc = T.tensor4()  # symbolic var for inputs for supervised loss
Yc = T.matrix()   # symbolic var for one-hot labels for supervised loss
Z0 = T.matrix()   # symbolic var for top-most latent variables for generator

# Gather symbolic outputs from inference with labeled data
print("Compiling and testing labeled inference...")
im_res_dict = inf_gen_model.apply_im_labeled(Xc, Yc, noise=bu_noise)
su_obs_vae_nlls = im_res_dict['obs_vae_nlls']
su_obs_vae_klds = im_res_dict['obs_vae_klds']
su_obs_cls_nlls = im_res_dict['obs_cls_nlls']
su_log_p_xIz = T.mean(im_res_dict['log_p_xIz'])
su_kld_a = T.mean(im_res_dict['kld_a'])
su_kld_y = T.mean(im_res_dict['kld_y'])
su_kld_z = T.mean(im_res_dict['kld_z'])
su_ent_y = T.mean(im_res_dict['ent_y'])
su_batch_y_prob = im_res_dict['batch_y_prob']
su_batch_ent_y = im_res_dict['batch_ent_y']

su_inf_outputs = [su_obs_vae_nlls, su_obs_vae_klds, su_obs_cls_nlls, su_kld_a, su_kld_z, su_ent_y]
su_inf_outputs_names = ['obs_vae_nlls', 'obs_vae_klds', 'obs_cls_nlls', 'kld_a', 'kld_z', 'ent_y']
func_labeled_inf = theano.function([Xc, Yc], su_inf_outputs)
x_in = train_transform(Xtr_su[0:nbatch,:])
y_in = Ytr_su[0:nbatch,:]
result = func_labeled_inf(x_in, y_in)
print("LABELED INFERENCE OUTPUTS:")
su_out_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for n, v in zip(su_inf_outputs_names, result)])
print(su_out_str)

# Gather symbolic outputs from inference with unlabeled data
print("Compiling and testing unlabeled inference...")
# quick test of the marginalized BU/TD inference process
im_res_dict = inf_gen_model.apply_im_unlabeled_2(Xg, noise=bu_noise)
un_obs_nlls = im_res_dict['obs_nlls']
un_obs_klds = im_res_dict['obs_klds']
un_log_p_xIz = T.mean(im_res_dict['log_p_xIz'])
un_kld_z = T.mean(im_res_dict['kld_z'])
un_kld_a = T.mean(im_res_dict['kld_a'])
un_kld_y = T.mean(im_res_dict['kld_y'])
un_ent_y = T.mean(im_res_dict['ent_y'])
un_batch_ent_y = im_res_dict['batch_ent_y']

un_inf_outputs = [un_obs_nlls, un_obs_klds, un_kld_a, un_kld_z, un_ent_y, un_batch_ent_y]
un_inf_outputs_names = ['obs_nlls', 'obs_klds', 'kld_a', 'kld_z', 'ent_y', 'batch_ent_y']
func_unlabeled_inf = theano.function([Xg], un_inf_outputs)
result = func_unlabeled_inf(train_transform(Xtr_un[0:nbatch,:]))
print("UNLABELED INFERENCE OUTPUTS:")
un_out_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for n, v in zip(un_inf_outputs_names, result)])
print(un_out_str)

#
# compiled functions (so far):
# ---------------------------
#   1. func_labeled_inf  : computes various outputs of inference for labeled data
#   2. func_unlabeled_inf: computes various outputs of inference for unlabeled data
#


#######################################################
# CONSTRUCT COST VARIABLES FOR THE TRAINING OBJECTIVE #
#######################################################

su_vae_nll = T.mean(su_obs_vae_nlls)
su_vae_kld = T.mean(su_obs_vae_klds)
su_cls_nll = T.mean(su_obs_cls_nlls)
su_cost = su_vae_nll + su_vae_kld + (lam_su_cls[0] * su_cls_nll)

un_vae_nll = T.mean(un_obs_nlls)
un_vae_kld = T.mean(un_obs_klds)
un_oent_y = un_ent_y
un_bent_y = un_batch_ent_y
un_cost = un_vae_nll + un_vae_kld + \
          (lam_obs_ent_y[0] * un_oent_y) + \
          (lam_batch_ent_y[0] * un_bent_y)

# The full cost requires inputs Xg, Xc, and Yc. Respectively, these are the
# unlabeled observations, the labeled observations, and the latter's labels.
reg_cost = 2e-5 * sum([T.sum(p**2.0) for p in all_params])
full_cost = (lam_su[0] * su_cost) + un_cost + reg_cost

# Collect costs that we should evaluate and track during training
train_outputs = [su_cost, su_vae_nll, su_vae_kld, su_cls_nll, un_cost,
                 un_vae_nll, un_kld_a, un_kld_z, un_oent_y, un_bent_y]
train_outputs_names = ['su_cost', 'su_vae_nll', 'su_vae_kld',
                       'su_cls_nll', 'un_cost', 'un_vae_nll',
                       'un_kld_a', 'un_kld_z', 'un_oent_y',
                       'un_bent_y']

#####################################################
# CONSTRUCT PARAMETER UPDATES AND TRAINING FUNCTION #
#####################################################

# stuff for performing updates
lrt = sharedX(0.0001)
b1t = sharedX(0.8)
param_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)

# # build training cost and update functions
t = time()
print("Computing gradients...")
param_updates, param_grads = param_updater(all_params, full_cost,
                                           return_grads=True)
print("Compiling training functions...")
# grab the model's sampling function
sample_func = inf_gen_model.generate_samples # takes inputs (z0, yi)
# compile functions for computing model costs and updates
func_train = theano.function([Xg, Xc, Yc], train_outputs, updates=param_updates)
func_train_costs = theano.function([Xg, Xc, Yc], train_outputs)
#func_train = func_train_costs

print("Testing training cost evaluation...")
# quick test of training function (without
xg = train_transform(Xtr_un[0:nbatch,:])
xc = train_transform(Xtr_su[0:nbatch,:])
yc = Ytr_su[0:nbatch,:]
result = func_train_costs(xg, xc, yc)

su_out_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                        n, v in zip(train_outputs_names, result) if (n.find('su_') > -1)])
un_out_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                        n, v in zip(train_outputs_names, result) if (n.find('un_') > -1)])
print("-- {}".format(su_out_str))
print("-- {}".format(un_out_str))

# compile a function for predicting y, and wrap it for multi-sample evaluation
y_softmax, y_unnorm = inf_gen_model.apply_predict_y(Xg, noise=bu_noise)
func_predict_y = theano.function([Xg], [y_softmax, y_unnorm])
def predict_y(xg, sample_count=10):
    y_sm, y_un = func_predict_y(xg)
    if sample_count > 1:
        for s in range(sample_count-1):
            ysm, yun = func_predict_y(xg)
            y_sm = y_sm + ysm
            y_un = y_un + yun
    return y_sm, y_un


# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

t = time()
ntrain = Xtr_un.shape[0]
batch_idx_un = np.arange(ntrain)
batch_idx_un = np.concatenate([batch_idx_un[:,np.newaxis],batch_idx_un[:,np.newaxis]], axis=1)
for epoch in range(1, niter+niter_decay+1):
    # shuffle training and validation data
    Xtr_un = shuffle(Xtr_un)
    Xtr_su, Ytr_su = shared_shuffle(Xtr_su, Ytr_su)
    Xva, Yva = shared_shuffle(Xva, Yva)
    # initialize cost arrays
    epoch_costs = [0. for i in range(len(train_outputs))]
    val_epoch_costs = [0. for i in range(len(train_outputs))]
    su_idx = 0
    val_idx = 0
    batch_count = 0.
    val_batch_count = 0.
    for bidx in tqdm(iter_data(batch_idx_un, size=nbatch), total=ntrain/nbatch):
        # get a batch of unlabeled data
        bidx = bidx[:,0].ravel()
        imb_x = Xtr_un[bidx,:].copy()
        # get a batch of labeled data
        su_idx = su_idx + nbatch
        if (su_idx + nbatch) > Xtr_su.shape[0]:
            Xtr_su, Ytr_su = shared_shuffle(Xtr_su, Ytr_su)
            su_idx = 0
        end_idx = su_idx + nbatch
        cmb_x = Xtr_su[su_idx:end_idx,:]
        cmb_y = Ytr_su[su_idx:end_idx,:]
        # transform training batches to "binary image format"
        imb_x_img = train_transform(binarize_data(imb_x))
        cmb_x_img = train_transform(binarize_data(cmb_x))
        # train vae on training batch
        result = func_train(imb_x_img, cmb_x_img, cmb_y)
        epoch_costs = [(v1 + v2) for v1, v2 in zip(result, epoch_costs)]
        batch_count += 1
        # evaluate a validation batch, if required
        if val_batch_count < 50:
            val_idx = val_idx + nbatch
            if (val_idx + nbatch) > Xva.shape[0]:
                val_idx = 0
            end_idx = val_idx + nbatch
            vmb_x = Xva[val_idx:end_idx,:].copy()
            vmb_y = Yva[val_idx:end_idx,:].copy()
            vmb_x_img = train_transform(binarize_data(vmb_x))
            result = func_train_costs(vmb_x_img, vmb_x_img, vmb_y)
            val_epoch_costs = [(v1 + v2) for v1, v2 in zip(result, val_epoch_costs)]
            val_batch_count += 1
    if (epoch == 50) or (epoch == 100) or (epoch == 150):
        # cut learning rate in half
        lr = lrt.get_value(borrow=False)
        lr = lr / 2.0
        lrt.set_value(floatX(lr))
        b1 = b1t.get_value(borrow=False)
        b1 = b1 + ((0.95 - b1) / 2.0)
        b1t.set_value(floatX(b1))
    if epoch > niter:
        # linearly decay learning rate
        lr = lrt.get_value(borrow=False)
        remaining_epochs = (niter + niter_decay + 1) - epoch
        lrt.set_value(floatX(lr - (lr / remaining_epochs)))
    ###################
    # SAVE PARAMETERS #
    ###################
    inf_gen_model.dump_params(inf_gen_param_file)
    ####################################
    # EVALUATE CLASSIFICATION ACCURACY #
    ####################################
    max_idx = min(Xtr_su.shape[0], 500)
    y_sm_su, y_un_su = predict_y(train_transform(Xtr_su[:max_idx,:]))
    tr_acc = class_accuracy(y_sm_su, Ytr_su[:max_idx,:])
    max_idx = min(Xva.shape[0], 500)
    y_sm_va, y_un_va = predict_y(train_transform(Xva[:max_idx,:]))
    va_acc = class_accuracy(y_sm_va, Yva[:max_idx,:])
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    epoch_costs = [(c / batch_count) for c in epoch_costs]
    val_epoch_costs = [(c / val_batch_count) for c in val_epoch_costs]
    str1 = "Epoch {}:".format(epoch)
    # training diagnostics
    tr_su_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                            n, v in zip(train_outputs_names, epoch_costs) if (n.find('su_') > -1)])
    tr_un_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                            n, v in zip(train_outputs_names, epoch_costs) if (n.find('un_') > -1)])
    tr_str1 = "    Train:"
    tr_str2 = "    -- {}".format(tr_su_str)
    tr_str3 = "    -- {}".format(tr_un_str)
    # validation diagnostics
    va_su_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                            n, v in zip(train_outputs_names, val_epoch_costs) if (n.find('su_') > -1)])
    va_un_str = ", ".join(["{0:s}: {1:.4f}".format(n, 1.0*np.mean(v)) for \
                            n, v in zip(train_outputs_names, val_epoch_costs) if (n.find('un_') > -1)])
    va_str1 = "    Valid:"
    va_str2 = "    -- {}".format(va_su_str)
    va_str3 = "    -- {}".format(va_un_str)
    acc_str = "    ACC: tr={0:.4f}, va={1:.4f}".format(tr_acc, va_acc)
    joint_str = "\n".join([str1, tr_str1, tr_str2, tr_str3,
                           va_str1, va_str2, va_str3, acc_str])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    # generate some samples from the model prior
    sample_yi = np.concatenate([np.eye(nyc) for i in range(nyc)], axis=0)
    sample_z0 = np.repeat(rand_gen(size=(nyc, nz0)), nyc, axis=0)
    samples = np.asarray(sample_func(floatX(sample_z0), floatX(sample_yi)))
    grayscale_grid_vis(draw_transform(samples), (nyc, nyc), "{}/gen1_{}.png".format(result_dir, epoch))
    sample_yi = np.concatenate([np.eye(nyc), np.eye(nyc)], axis=0)
    sample_z0 = rand_gen(size=(sample_yi.shape[0], nz0))
    sample_yi = np.repeat(sample_yi, 20, axis=0)
    sample_z0 = np.repeat(sample_z0, 20, axis=0)
    samples = np.asarray(sample_func(floatX(sample_z0), floatX(sample_yi)))
    grayscale_grid_vis(draw_transform(samples), (2*nyc, 20), "{}/gen2_{}.png".format(result_dir, epoch))






##############
# EYE BUFFER #
##############
