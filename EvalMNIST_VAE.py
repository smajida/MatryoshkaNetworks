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
from lib.data_utils import shuffle, iter_data
from load import load_binarized_mnist

#
# Phil's business
#
from MatryoshkaModules import BasicConvModule, GenConvResModule, \
                              GenFCModule, InfConvMergeModule, \
                              InfFCModule, BasicConvResModule, \
                              DiscConvResModule, DiscFCModule
from MatryoshkaNetworks import InfGenModel, DiscNetworkGAN, GenNetworkGAN

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_vae_lrelu_mods_2abc_4bc_no_bn_iwae_50x15'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load binarized MNIST dataset
data_path = "{}/data/".format(EXP_DIR)
Xtr, Xva, Xte = load_binarized_mnist(data_path=data_path)
Xva = Xte[:,:]


set_seed(1)       # seed for shared rngs
nc = 1            # # of channels in image
nbatch = 2       # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 32          # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngf = 32          # base # of filters for conv layers in generative stuff
ngfc = 128        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
multi_rand = True # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_bn = False     # whether to use batch normalization throughout the model
use_td_cond = False # whether to use top-down conditioning in generator
act_func = 'lrelu'
iwae_samples = 1500  # number of samples to use in MEN bound

ntrain = Xtr.shape[0]

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
    rand_dim=nz0,
    out_shape=(ngf*4, 7, 7),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2a = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=1,
    mod_name='td_mod_2a'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2b = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=1,
    mod_name='td_mod_2b'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2c = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=1,
    mod_name='td_mod_2c'
) # output is (batch, ngf*2, 7, 7)

# (7, 7) -> (14, 14)
td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*2, 14, 14)

## (14, 14) -> (14, 14)
#td_module_4a = \
#GenConvResModule(
#    in_chans=(ngf*2),
#    out_chans=(ngf*2),
#    conv_chans=(ngf*2),
#    rand_chans=nz1,
#    filt_shape=(3,3),
#    use_rand=multi_rand,
#    use_conv=use_conv,
#    apply_bn=use_bn,
#    act_func=act_func,
#    us_stride=1,
#    mod_name='td_mod_4a'
#) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (14, 14)
td_module_4b = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=1,
    mod_name='td_mod_4b'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (28, 28)
td_module_4c = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=2,
    mod_name='td_mod_4c'
) # output is (batch, ngf*1, 28, 28)

# (28, 28) -> (28, 28)
td_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_5'
) # output is (batch, c, 28, 28)

# modules must be listed in "evaluation order"
td_modules = [td_module_1, td_module_2a, td_module_2b, td_module_2c,
              td_module_3, td_module_4b, td_module_4c, td_module_5]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

# (7, 7) -> FC
bu_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*7*7),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
) # output is (batch, nz0), (batch, nz0)

# (7, 7) -> (7, 7)
bu_module_2a = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2a'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
bu_module_2b = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2b'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
bu_module_2c = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2c'
) # output is (batch, ngf*4, 7, 7)

# (14, 14) -> (7, 7)
bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 7, 7)

## (14, 14) -> (14, 14)
#bu_module_4a = \
#BasicConvResModule(
#    in_chans=(ngf*2),
#    out_chans=(ngf*2),
#    conv_chans=(ngf*2),
#    filt_shape=(3,3),
#    use_conv=use_conv,
#    apply_bn=use_bn,
#    stride='single',
#    act_func=act_func,
#    mod_name='bu_mod_4a'
#) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (14, 14)
bu_module_4b = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_4b'
) # output is (batch, ngf*2, 14, 14)

# (28, 28) -> (14, 14)
bu_module_4c = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_4c'
) # output is (batch, ngf*2, 14, 14)

# (28, 28) -> (28, 28)
bu_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=False,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_6'
) # output is (batch, ngf*1, 28, 28)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_5, bu_module_4c, bu_module_4b,
              bu_module_3, bu_module_2c, bu_module_2b, bu_module_2a,
              bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_2a = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_2a'
) # merge input to td_mod_2a and output of bu_mod_2a, to place a distribution
  # over the rand_vals used in td_mod_2a.

im_module_2b = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_2b'
) # merge input to td_mod_2b and output of bu_mod_2b, to place a distribution
  # over the rand_vals used in td_mod_2b.

im_module_2c = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_2c'
) # merge input to td_mod_2c and output of bu_mod_2c, to place a distribution
  # over the rand_vals used in td_mod_2c.

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
) # merge input to td_mod_3 and output of bu_mod_3, to place a distribution
  # over the rand_vals used in td_mod_3.

#im_module_4a = \
#InfConvMergeModule(
#    td_chans=(ngf*2),
#    bu_chans=(ngf*2),
#    rand_chans=nz1,
#    conv_chans=(ngf*2),
#    use_conv=True,
#    apply_bn=use_bn,
#    use_td_cond=use_td_cond,
#    act_func=act_func,
#    mod_name='im_mod_4a'
#) # merge input to td_mod_4 and output of bu_mod_4, to place a distribution
#  # over the rand_vals used in td_mod_4.

im_module_4b = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_4b'
) # merge input to td_mod_4 and output of bu_mod_4, to place a distribution
  # over the rand_vals used in td_mod_4.

im_module_4c = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    act_func=act_func,
    mod_name='im_mod_4c'
) # merge input to td_mod_4 and output of bu_mod_4, to place a distribution
  # over the rand_vals used in td_mod_4.

im_modules = [im_module_2a, im_module_2b, im_module_2c, im_module_3,
              im_module_4b, im_module_4c]

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
    'td_mod_2a': {'bu_module': 'bu_mod_2a', 'im_module': 'im_mod_2a'},
    'td_mod_2b': {'bu_module': 'bu_mod_2b', 'im_module': 'im_mod_2b'},
    'td_mod_2c': {'bu_module': 'bu_mod_2c', 'im_module': 'im_mod_2c'},
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
#    'td_mod_4a': {'bu_module': 'bu_mod_4a', 'im_module': 'im_mod_4a'},
    'td_mod_4b': {'bu_module': 'bu_mod_4b', 'im_module': 'im_mod_4b'},
    'td_mod_4c': {'bu_module': 'bu_mod_4c', 'im_module': 'im_mod_4c'}
}

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=sigmoid
)
###################
# LOAD PARAMETERS #
###################
inf_gen_model.load_params(inf_gen_param_file)

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

######################
# Compute IWAE bound #
######################
# run an inference and reconstruction pass through the generative stuff
batch_size = Xg.shape[0]
Xg_rep = T.extra_ops.repeat(Xg, iwae_samples, axis=0)
im_res_dict = inf_gen_model.apply_im(Xg_rep)
Xg_rep_recon = im_res_dict['td_output']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

log_p_x = T.sum(log_prob_bernoulli( \
                T.flatten(Xg_rep,2), T.flatten(Xg_rep_recon,2),
                do_sum=False), axis=1)

# compute quantities used in the IWAE bound
log_ws_vec = log_p_x + log_p_z - log_q_z
log_ws_mat = log_ws_vec.reshape((batch_size, iwae_samples))
ws_mat = log_ws_mat - T.max(log_ws_mat, axis=1, keepdims=True)
ws_mat = T.exp(ws_mat)
nis_weights = ws_mat / T.sum(ws_mat, axis=1, keepdims=True)
nis_weights = theano.gradient.disconnected_grad(nis_weights)

iwae_obs_costs = -1.0 * (T.sum((nis_weights * log_ws_mat), axis=1) - \
                         T.sum((nis_weights * T.log(nis_weights)), axis=1))

iwae_bound = T.mean(iwae_obs_costs)
iwae_bound_lme = -1.0 * T.mean(log_mean_exp(log_ws_mat, axis=1))

########################################
# Compute VAE bound using same samples #
########################################
# compute a VAE-style reconstruction cost averaged over IWAE samples
vae_obs_nlls = -1.0 * T.mean(log_p_x.reshape((batch_size, iwae_samples)), axis=1)
vae_nll_cost = T.mean(vae_obs_nlls)
# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dict.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]
# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([T.mean(mod_kld.reshape((batch_size, iwae_samples)), axis=1) \
                     for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

vae_bound = vae_nll_cost + vae_kld_cost

######################################################
# Get functions for free sampling and reconstruction #
######################################################
# get simple reconstruction, for other purposes
im_rd = inf_gen_model.apply_im(Xg)
Xg_recon = im_rd['td_output']
# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)


# build training cost and update functions
t = time()
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr[0:100,:])) # cheeky model implementation test
print("Compiling cost computing functions...")
# collect costs for generator parameters
g_basic_costs = [iwae_bound, vae_bound, vae_nll_cost, vae_kld_cost,
                 iwae_bound_lme]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['iwae_bound', 'vae_bound', 'vae_nll_cost', 'vae_kld_cost',
              'iwae_bound_lme']
# compile function for computing generator costs and updates
g_eval_func = theano.function([Xg], g_basic_costs)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/EVAL.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

Xva_blocks = np.split(Xva, 10, axis=0)
for epoch in range(5):
    for block_num, Xva_block in enumerate(Xva_blocks):
        Xva_block = shuffle(Xva_block)
        obs_count = Xva_block.shape[0]
        g_epoch_costs = [0. for c in g_basic_costs]
        g_batch_count = 0.
        for imb in tqdm(iter_data(Xva_block, size=nbatch), total=obs_count/nbatch):
            # transform validation batch to "image format"
            imb_img = train_transform(imb)
            # train vae on training batch
            g_result = g_eval_func(imb_img.astype(theano.config.floatX))
            g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
            g_batch_count += 1
        ##################################
        # QUANTITATIVE DIAGNOSTICS STUFF #
        ##################################
        g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
        str1 = "Epoch {}, block {}:".format(epoch, block_num)
        g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                     for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
        str2 = "    {}".format(" ".join(g_bc_strs))
        joint_str = "\n".join([str1, str2])
        print(joint_str)
        out_file.write(joint_str+"\n")
        out_file.flush()
        ######################
        # DRAW SOME PICTURES #
        ######################
        sample_z0mb = np.repeat(rand_gen(size=(20, nz0)), 20, axis=0)
        samples = np.asarray(sample_func(sample_z0mb))
        grayscale_grid_vis(draw_transform(samples), (20, 20), "{}/eval_gen_e{}_b{}.png".format(result_dir, epoch, block_num))







##############
# EYE BUFFER #
##############
