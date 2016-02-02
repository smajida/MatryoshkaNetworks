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
from lib.ops import log_mean_exp
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
                              DiscConvResModule, DiscFCModule, MlpFCModule
from MatryoshkaNetworks import InfGenModel, SimpleMLP

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
sup_count = 100
desc = "test_ss_relu_{}_labels".format(sup_count)
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load binarized MNIST dataset
data_path = "{}/data/mnist.pkl.gz".format(EXP_DIR)
data_dict = load_udm_ss(data_path, sup_count, im_dim=32)
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


set_seed(1)       # seed for shared rngs
sup_count = 100   # number of labeled examples
nc = 1            # # of channels in image
nbatch = 100      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz0 = 32          # # of dim for Z0
nz0_c = 16        # # of dim in Z0 to use for classifying
nz1 = 16          # # of dim for Z1
ngf = 32          # base # of filters for conv layers in generative stuff
ngfc = 128        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
niter = 200       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_bn = True     # whether to use batch normalization throughout the model
use_td_cond = False # whether to use top-down conditioning in generator
act_func = 'lrelu' # activation func to use where they can be selected
grad_noise = 0.04 # initial noise for the gradients

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

# FC -> (2, 2)
td_module_1 = \
GenFCModule(
    rand_dim=nz0,
    out_shape=(ngf*8, 2, 2),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=use_bn,
    mod_name='td_mod_1'
) # output is (batch, ngf*8, 2, 2)

# (2, 2) -> (4, 4)
td_module_2 = \
GenConvResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 4, 4)

# (4, 4) -> (8, 8)
td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*4, 8, 8)

# (8, 8) -> (16, 16)
td_module_4 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_4'
) # output is (batch, ngf*2, 16, 16)

# (16, 16) -> (32, 32)
td_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    us_stride=2,
    mod_name='td_mod_5'
) # output is (batch, ngf*1, 32, 32)

# (32, 32) -> (32, 32)
td_module_6 = \
BasicConvModule(
    in_chans=(ngf*1),
    out_chans=nc,
    filt_shape=(3,3),
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_6'
) # output is (batch, c, 32, 32)

# modules must be listed in "evaluation order"
td_modules = [td_module_1, td_module_2, td_module_3,
              td_module_4, td_module_5, td_module_6]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

# (2, 2) -> FC
bu_module_1 = \
InfFCModule(
    bu_chans=(ngf*8*2*2),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    unif_drop=0.2,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
) # output is (batch, nz0), (batch, nz0)

# (4, 4) -> (2, 2)
bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*8),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.2,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_2'
) # output is (batch, ngf*8, 2, 2)

# (8, 8) -> (4, 4)
bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.2,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 4, 4)

# (16, 16) -> (8, 8)
bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.2,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_4'
) # output is (batch, ngf*4, 8, 8)

# (32, 32) -> (16, 16)
bu_module_5 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    use_conv=use_conv,
    chan_drop=0.2,
    unif_drop=0.0,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_5'
) # output is (batch, ngf*2, 16, 16)

# (32, 32) -> (32, 32)
bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    chan_drop=0.2,
    unif_drop=0.0,
    apply_bn=False,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_6'
) # output is (batch, ngf*1, 32, 32)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_6, bu_module_5, bu_module_4,
              bu_module_3, bu_module_2, bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_2 = \
InfConvMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8),
    rand_chans=nz1,
    conv_chans=(ngf*4),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_2'
)

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_3'
)

im_module_4 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_4'
)

im_module_5 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_5'
)

im_modules = [im_module_2, im_module_3, im_module_4, im_module_5]

#
# Setup a description for where to get conditional distributions from. When
# there's no info here for a particular top-down module, we won't pass any
# random variables explicitly into the module, which will cause the module to
# generate its own random variables (unconditionally). When a "bu_module" is
# provided and an "im_module" is not, the conditional distribution is specified
# directly by the bu_module's output, and no merging (via an im_module) is
# required. This probably only happens at the "top" of the generator.
#
merge_info = {
    'td_mod_1': {'bu_module': 'bu_mod_1', 'im_module': None},
    'td_mod_2': {'bu_module': 'bu_mod_2', 'im_module': 'im_mod_2'},
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
    'td_mod_4': {'bu_module': 'bu_mod_4', 'im_module': 'im_mod_4'},
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'}
}

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=sigmoid
)

######################
# Setup a classifier #
######################

# cls_module_1 = \
# MlpFCModule(
#     in_dim=nz0_c,
#     out_dim=64,
#     apply_bn=False,
#     unif_drop=0.0,
#     act_func='lrelu',
#     use_bn_params=True,
#     mod_name='cls_mod_1'
# )
#
# cls_module_2 = \
# MlpFCModule(
#     in_dim=ngfc,
#     out_dim=nyc,
#     apply_bn=True,
#     unif_drop=0.0,
#     act_func='ident',
#     use_bn_params=True,
#     mod_name='cls_mod_2'
# )
#
# cls_modules = [cls_module_1, cls_module_2]

cls_module_1 = \
MlpFCModule(
    in_dim=nz0_c,
    out_dim=nyc,
    apply_bn=False,
    unif_drop=0.0,
    act_func='ident',
    use_bn_params=True,
    mod_name='cls_mod_1'
)

cls_modules = [ cls_module_1 ]

class_model = SimpleMLP(modules=cls_modules)


####################################
# Setup the optimization objective #
####################################
lam_vae = sharedX(floatX(np.asarray([1.0])))
lam_cls = sharedX(floatX(np.asarray([1.0])))
lam_cls_cls = sharedX(floatX(np.asarray([1.0])))
gen_params = inf_gen_model.gen_params
inf_params = inf_gen_model.inf_params
cls_params = class_model.params
all_params = gen_params + inf_params + cls_params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs for generative loss
Xc = T.tensor4()  # symbolic var for inputs for classification loss
Yc = T.matrix()   # symbolic vae for one-hot labels for classification loss
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generator

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# run an inference and reconstruction pass through the generative stuff
im_res_dict = inf_gen_model.apply_im(Xg)
Xg_recon = im_res_dict['td_output']
vae_kld_dict = im_res_dict['kld_dict']
vae_log_p_x = T.sum(log_prob_bernoulli( \
                    T.flatten(Xg,2), T.flatten(Xg_recon,2),
                    do_sum=False), axis=1)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * vae_log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# compute per-layer KL-divergence part of cost
vae_kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) \
                  for mod_name, mod_kld in vae_kld_dict.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) \
                  for mod_name, mod_kld in vae_kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in vae_kld_tuples]
# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in vae_kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

# combined cost for free-energy stuff
vae_cost = vae_nll_cost + vae_kld_cost

##########################################################
# CONSTRUCT COST VARIABLES FOR THE CLS PART OF OBJECTIVE #
##########################################################
# run an inference and reconstruction pass through the generative stuff
im_res_dict = inf_gen_model.apply_im(Xc)
Xc_recon = im_res_dict['td_output']
cls_kld_dict = im_res_dict['kld_dict']
cls_z_dict = im_res_dict['z_dict']
cls_z_top = cls_z_dict['td_mod_1']

# apply classifier to the top-most set of latent variables, maybe good idea?
Yc_recon = T.nnet.softmax( class_model.apply(cls_z_top[:,:nz0_c]) )

# compute reconstruction error part of free-energy
cls_log_p_x = T.sum(log_prob_bernoulli( \
                    T.flatten(Xc,2), T.flatten(Xc_recon,2),
                    do_sum=False), axis=1)
cls_obs_nlls = -1.0 * cls_log_p_x
cls_nll_cost = T.mean(cls_obs_nlls)

# compute per-layer KL-divergence part of cost
cls_kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) \
                  for mod_name, mod_kld in cls_kld_dict.items()]
cls_layer_klds = T.as_tensor_variable([T.mean(mod_kld) \
                  for mod_name, mod_kld in cls_kld_tuples])
cls_layer_names = [mod_name for mod_name, mod_kld in cls_kld_tuples]
# compute total per-observation KL-divergence part of cost
cls_obs_klds = sum([mod_kld for mod_name, mod_kld in cls_kld_tuples])
cls_kld_cost = T.mean(cls_obs_klds)

# combined cost for generator stuff
cls_cls_cost = lam_cls_cls[0] * T.mean(T.nnet.categorical_crossentropy(Yc_recon, Yc))
cls_cost = cls_nll_cost + cls_kld_cost + cls_cls_cost + \
           (T.mean(Yc**2.0) * T.sum(cls_z_dict['td_mod_1']**2.0))

#################################################################
# COMBINE VAE AND CLS OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################
# parameter regularization part of cost
reg_cost = 2e-5 * sum([T.sum(p**2.0) for p in all_params])
# full joint cost -- a weighted combination of free-energy and classification
full_cost = (lam_vae[0] * vae_cost) + (lam_cls[0] * cls_cost) + reg_cost

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

# stuff for performing updates
lrt = sharedX(0.0002)
b1t = sharedX(0.8)
gen_updater = updates.FuzzyAdam(lr=lrt, b1=b1t, b2=0.98, e=1e-4,
                                n=grad_noise, clipnorm=1000.0)
inf_updater = updates.FuzzyAdam(lr=lrt, b1=b1t, b2=0.98, e=1e-4,
                                n=grad_noise, clipnorm=1000.0)
cls_updater = updates.FuzzyAdam(lr=lrt, b1=b1t, b2=0.98, e=1e-4,
                                n=grad_noise, clipnorm=1000.0)


# build training cost and update functions
t = time()
print("Computing gradients...")
gen_updates, gen_grads = gen_updater(gen_params, full_cost, return_grads=True)
inf_updates, inf_grads = inf_updater(inf_params, full_cost, return_grads=True)
cls_updates, cls_grads = inf_updater(cls_params, full_cost, return_grads=True)
all_updates = gen_updates + inf_updates + cls_updates
gen_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in gen_grads]))
inf_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in inf_grads]))
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr_un[0:100,:]))
print("Compiling training functions...")
# collect costs for generator parameters
train_costs = [full_cost, vae_cost, cls_cost, vae_nll_cost, vae_kld_cost,
               cls_nll_cost, cls_kld_cost, cls_cls_cost, vae_layer_klds]

tc_idx = range(0, len(train_costs))
tc_names = ['full_cost', 'vae_cost', 'cls_cost', 'vae_nll_cost',
              'vae_kld_cost', 'cls_nll_cost', 'cls_kld_cost',
              'cls_cls_cost', 'vae_layer_klds']
# compile functions for computing generator costs and updates
cost_func = theano.function([Xg, Xc, Yc], train_costs)
pred_func = theano.function([Xc], Yc_recon)
train_func = theano.function([Xg, Xc, Yc], train_costs, updates=all_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
ntrain = Xtr_un.shape[0]
batch_idx_un = np.arange(ntrain)
batch_idx_un = np.concatenate([batch_idx_un[:,np.newaxis],batch_idx_un[:,np.newaxis]], axis=1)
for epoch in range(1, niter+niter_decay+1):
    Xtr_un = shuffle(Xtr_un)
    # set gradient noise
    eg_noise_ary = (grad_noise / np.sqrt(float(epoch)/2.0)) + np.zeros((1,))
    gen_updater.n.set_value(floatX(eg_noise_ary))
    inf_updater.n.set_value(floatX(eg_noise_ary))
    # dset relative weights of objectives
    lam_vae.set_value(floatX(np.asarray([1.0])))
    lam_cls.set_value(floatX(np.asarray([0.2])))
    lam_cls_cls.set_value(floatX(np.asarray([20.0])))
    # initialize cost arrays
    epoch_costs = [0. for i in range(8)]
    val_epoch_costs = [0. for i in range(8)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    batch_count = 0.
    val_batch_count = 0.
    train_acc = 0.
    val_acc = 0.
    for bidx in tqdm(iter_data(batch_idx_un, size=nbatch), total=ntrain/nbatch):
        # grab a validation batch, if required
        if val_batch_count < 50:
            start_idx = int(val_batch_count)*nbatch
            vmb_x = Xva[start_idx:(start_idx+nbatch),:].copy()
            vmb_y = Yva[start_idx:(start_idx+nbatch),:].copy()
        else:
            vmb_x = Xva[0:nbatch,:].copy()
            vmb_y = Yva[0:nbatch,:].copy()
        # transform training batch to "image format"
        bidx = bidx[:,0].ravel()
        imb_x = Xtr_un[bidx,:].copy()
        cmb_x = Xtr_su[0:100,:].copy()
        cmb_y = Ytr_su[0:100,:].copy()
        imb_x_img = train_transform(imb_x)
        cmb_x_img = train_transform(cmb_x)
        vmb_x_img = train_transform(vmb_x)
        # train vae on training batch
        result = train_func(imb_x_img, cmb_x_img, cmb_y)
        epoch_costs = [(v1 + v2) for v1, v2 in zip(result[:8], epoch_costs)]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(result[8], epoch_layer_klds)]
        y_pred = pred_func(cmb_x_img)
        train_acc += class_accuracy(y_pred, cmb_y, return_raw_count=False)
        batch_count += 1
        # evaluate vae on validation batch
        if val_batch_count < 25:
            val_result = train_func(vmb_x_img, vmb_x_img, vmb_y)
            val_epoch_costs = [(v1 + v2) for v1, v2 in zip(val_result[:8], val_epoch_costs)]
            y_pred = pred_func(vmb_x_img)
            val_acc += class_accuracy(y_pred, vmb_y, return_raw_count=False)
            val_batch_count += 1
    if (epoch == 20) or (epoch == 50) or (epoch == 100) or (epoch == 200):
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
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    epoch_costs = [(c / batch_count) for c in epoch_costs]
    epoch_layer_klds = [(c / batch_count) for c in epoch_layer_klds]
    train_acc = train_acc / batch_count
    val_epoch_costs = [(c / val_batch_count) for c in val_epoch_costs]
    val_acc = val_acc / val_batch_count
    str1 = "Epoch {}:".format(epoch)
    tc_strs = ["{0:s}: {1:.2f},".format(c_name, epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(tc_idx[:8], tc_names[:8])]
    str2 = " ".join(tc_strs)
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str3 = "    module kld -- {}".format(" ".join(kld_strs))
    str4 = "    train_acc: {0:.4f}, val_acc: {1:.4f}, val_cls_cls_cost: {2:.4f}".format( \
            train_acc, val_acc, val_epoch_costs[7])
    joint_str = "\n".join([str1, str2, str3, str4])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    # generate some samples from the model prior
    sample_z0mb = np.repeat(rand_gen(size=(20, nz0)), 20, axis=0)
    samples = np.asarray(sample_func(sample_z0mb))
    grayscale_grid_vis(draw_transform(samples), (20, 20), "{}/gen_{}.png".format(result_dir, epoch))
    # test reconstruction performance (inference + generation)
    tr_rb = Xtr_un[0:100,:]
    va_rb = Xva[0:100,:]
    # get the model reconstructions
    tr_rb = train_transform(tr_rb)
    va_rb = train_transform(va_rb)
    tr_recons = recon_func(tr_rb)
    va_recons = recon_func(va_rb)
    # stripe data for nice display (each reconstruction next to its target)
    tr_vis_batch = np.zeros((200, nc, npx, npx))
    va_vis_batch = np.zeros((200, nc, npx, npx))
    for rec_pair in range(100):
        idx_in = 2*rec_pair
        idx_out = 2*rec_pair + 1
        tr_vis_batch[idx_in,:,:,:] = tr_rb[rec_pair,:,:,:]
        tr_vis_batch[idx_out,:,:,:] = tr_recons[rec_pair,:,:,:]
        va_vis_batch[idx_in,:,:,:] = va_rb[rec_pair,:,:,:]
        va_vis_batch[idx_out,:,:,:] = va_recons[rec_pair,:,:,:]
    # draw images...
    grayscale_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(result_dir, epoch))
    grayscale_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(result_dir, epoch))







##############
# EYE BUFFER #
##############
