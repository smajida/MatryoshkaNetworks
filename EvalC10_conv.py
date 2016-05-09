import os
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm

import sys
import theano
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.ops import log_mean_exp, binarize_data, fuzz_data
from lib.costs import log_prob_bernoulli, log_prob_gaussian
from lib.vis import grayscale_grid_vis, color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import load_binarized_mnist, load_udm, load_cifar10

#
# Phil's business
#
from MatryoshkaModules import BasicConvModule, GenTopModule, InfTopModule, \
                              GenConvPertModule, BasicConvPertModule, \
                              GenConvGRUModule, InfConvMergeModuleIMS
from MatryoshkaNetworks import InfGenModel

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = './cifar10'

# setup paths for dumping diagnostic info
desc = 'test_conv_baby_steps_white_input'
result_dir = '{}/results/{}'.format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load CIFAR 10 dataset
data_path = '{}/data/'.format(EXP_DIR)
Xtr, Ytr, Xva, Yva = load_cifar10(data_path, va_split=None, dtype='float32',
                                  grayscale=False)


set_seed(123)       # seed for shared rngs
nc = 3              # # of channels in image
nbatch = 25         # # of examples in batch
npx = 32            # # of pixels width/height of images
nz0 = 32            # # of dim for Z0
nz1 = 4             # # of dim for Z1
ngf = 32            # base # of filters for conv layers in generative stuff
ngfc = 128          # # of filters in fully connected layers of generative stuff
nx = npx * npx * nc   # # of dimensions in X
niter = 150         # # of iter at starting learning rate
niter_decay = 250   # # of iter to linearly decay learning rate to zero
multi_rand = True   # whether to use stochastic variables at multiple scales
use_conv = True     # whether to use "internal" conv layers in gen/disc networks
use_bn = False      # whether to use batch normalization throughout the model
act_func = 'lrelu'  # activation func to use where they can be selected
noise_std = 0.0     # amount of noise to inject in BU and IM modules
use_bu_noise = False
use_td_noise = False
iwae_samples = 10
inf_mt = 0
use_td_cond = False
depth_4x4 = 3
depth_8x8 = 3
depth_16x16 = 3

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xtr.shape[0]


def train_transform(X, add_fuzz=True):
    # transform vectorized observations into convnet inputs
    X = X * 255.  # scale X to be in [0, 255]
    if add_fuzz:
        X = fuzz_data(X, scale=1., rand_type='uniform')
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3))


def draw_transform(X):
    # transform vectorized observations into drawable greyscale images
    X = X * 1.  # 255.0
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1))


def rand_gen(size, noise_type='normal'):
    if noise_type == 'normal':
        r_vals = floatX(np_rng.normal(size=size))
    elif noise_type == 'uniform':
        r_vals = floatX(np_rng.uniform(size=size, low=-1.0, high=1.0))
    else:
        assert False, "unrecognized noise type!"
    return r_vals


def estimate_gauss_params(X, samples=10):
    # compute data mean
    mu = np.mean(X, axis=0, keepdims=True) + 0.5
    Xc = X - mu

    # compute data covariance
    C = np.zeros((X.shape[1], X.shape[1]))
    for i in range(samples):
        Xc_i = fuzz_data(Xc, scale=1., rand_type='uniform')
        C = C + (np.dot(Xc_i.T, Xc_i) / Xc_i.shape[0])
    C = C / float(samples)
    return mu, C


def estimate_whitening_transform(X, samples=10):
    # estimate a whitening transform (mean shift and whitening matrix)
    # for the data in X, with added 'dequantization' noise.
    mu, C = estimate_gauss_params(X, samples=samples)

    # get eigenvalues and eigenvectors of covariance
    U, S, V = np.linalg.svd(C)
    # compute whitening matrix
    D = np.dot(U, np.diag(1. / np.sqrt(S + 1e-5)))
    W = np.dot(D, U.T)
    return W, mu


def nats2bpp(nats):
    bpp = (nats / (npx * npx * nc)) / np.log(2.)
    return bpp


def iwae_multi_eval(x, iters, cost_func, iwae_num):
    # slow multi-pass evaluation of IWAE bound.
    log_p_x = []
    log_p_z = []
    log_q_z = []
    for i in range(iters):
        result = cost_func(x)
        b_size = int(result[0].shape[0] / iwae_num)
        log_p_x.append(result[0].reshape((b_size, iwae_num)))
        log_p_z.append(result[1].reshape((b_size, iwae_num)))
        log_q_z.append(result[2].reshape((b_size, iwae_num)))
    # stack up results from multiple passes
    log_p_x = np.concatenate(log_p_x, axis=1)
    log_p_z = np.concatenate(log_p_z, axis=1)
    log_q_z = np.concatenate(log_q_z, axis=1)
    # compute the IWAE bound for each example in x
    log_ws_mat = log_p_x + log_p_z - log_q_z
    iwae_bounds = -1.0 * np_log_mean_exp(log_ws_mat, axis=1)
    return iwae_bounds


tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

#########################################
# Setup the top-down processing modules #
# -- these do generation                #
#########################################

# FC -> (7, 7)
td_module_1 = \
    GenTopModule(
        rand_dim=nz0,
        out_shape=(ngf * 4, 8, 8),
        fc_dim=ngfc,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='td_mod_1')

# grow the (4, 4) -> (4, 4) part of network
td_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'td_mod_2{}'.format(alphabet[i])
    new_module = \
        GenConvPertModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 4),
            conv_chans=(ngf * 4),
            rand_chans=nz1,
            filt_shape=(3, 3),
            use_rand=multi_rand,
            use_conv=use_conv,
            apply_bn=use_bn,
            act_func=act_func,
            us_stride=1,
            mod_name=mod_name)
    td_modules_4x4.append(new_module)
# manual stuff for parameter sharing....

# (4, 4) -> (8, 8)
td_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 4),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='td_mod_3')

# grow the (8, 8) -> (8, 8) part of network
td_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'td_mod_4{}'.format(alphabet[i])
    new_module = \
        GenConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            rand_chans=nz1,
            filt_shape=(3, 3),
            use_rand=multi_rand,
            use_conv=use_conv,
            apply_bn=use_bn,
            act_func=act_func,
            us_stride=1,
            mod_name=mod_name)
    td_modules_8x8.append(new_module)

# (8, 8) -> (16, 16)
td_module_5 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='td_mod_5')

# grow the (16, 16) -> (16, 16) part of network
td_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'td_mod_6{}'.format(alphabet[i])
    new_module = \
        GenConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            rand_chans=nz1,
            filt_shape=(3, 3),
            use_rand=multi_rand,
            use_conv=use_conv,
            apply_bn=use_bn,
            act_func=act_func,
            us_stride=1,
            mod_name=mod_name)
    td_modules_16x16.append(new_module)
# manual stuff for parameter sharing....

# (16, 16) -> (32, 32)
td_module_7 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 2),
        out_chans=(ngf * 1),
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name='td_mod_7')

# (32, 32) -> (32, 32)
td_module_8 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 1),
        out_chans=nc,
        apply_bn=False,
        rescale_output=True,
        use_noise=False,
        stride='single',
        act_func='ident',
        mod_name='td_mod_8')

# modules must be listed in "evaluation order"
td_modules = [td_module_1] + \
             td_modules_4x4 + \
             [td_module_3] + \
             td_modules_8x8 + \
             [td_module_5] + \
             td_modules_16x16 + \
             [td_module_7, td_module_8]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

# (4, 4) -> FC
bu_module_1 = \
    InfTopModule(
        bu_chans=(ngf * 4 * 8 * 8),
        fc_chans=ngfc,
        rand_chans=nz0,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='bu_mod_1')

# grow the (4, 4) -> (4, 4) part of network
bu_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'bu_mod_2{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 4),
            conv_chans=(ngf * 4),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_4x4.append(new_module)
bu_modules_4x4.reverse()

# (8, 8) -> (4, 4)
bu_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 4),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_3')

# grow the (8, 8) -> (8, 8) part of network
bu_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'bu_mod_4{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_8x8.append(new_module)
bu_modules_8x8.reverse()

# (8, 8) -> (16, 16)
bu_module_5 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_5')

# grow the (16, 16) -> (16, 16) part of network
bu_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'bu_mod_6{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_16x16.append(new_module)
bu_modules_16x16.reverse()

# (16, 16) -> (32, 32)
bu_module_7 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 1),
        out_chans=(ngf * 2),
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name='bu_mod_7')

# (32, 32) -> (32, 32)
bu_module_8 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=nc,
        out_chans=(ngf * 1),
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name='bu_mod_8')

# modules must be listed in "evaluation order"
bu_modules = [bu_module_8, bu_module_7] + \
             bu_modules_16x16 + \
             [bu_module_5] + \
             bu_modules_8x8 + \
             [bu_module_3] + \
             bu_modules_4x4 + \
             [bu_module_1]


#########################################
# Setup the information merging modules #
#########################################

# FC -> (4, 4)
im_module_1 = \
    GenTopModule(
        rand_dim=nz0,
        out_shape=(ngf * 4, 8, 8),
        fc_dim=ngfc,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='im_mod_1')

# grow the (4, 4) -> (4, 4) part of network
im_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'im_mod_2{}'.format(alphabet[i])
    new_module = \
        InfConvMergeModuleIMS(
            td_chans=(ngf * 4),
            bu_chans=(ngf * 4),
            im_chans=(ngf * 4),
            rand_chans=nz1,
            conv_chans=(ngf * 4),
            use_conv=True,
            use_td_cond=use_td_cond,
            apply_bn=use_bn,
            mod_type=inf_mt,
            act_func=act_func,
            mod_name=mod_name)
    im_modules_4x4.append(new_module)

# (4, 4) -> (8, 8)
im_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 4),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='im_mod_3')

# grow the (8, 8) -> (8, 8) part of network
im_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'im_mod_4{}'.format(alphabet[i])
    new_module = \
        InfConvMergeModuleIMS(
            td_chans=(ngf * 2),
            bu_chans=(ngf * 2),
            im_chans=(ngf * 2),
            rand_chans=nz1,
            conv_chans=(ngf * 2),
            use_conv=True,
            use_td_cond=use_td_cond,
            apply_bn=use_bn,
            mod_type=inf_mt,
            act_func=act_func,
            mod_name=mod_name)
    im_modules_8x8.append(new_module)

# (8, 8) -> (16, 16)
im_module_5 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='im_mod_5')

# grow the (16, 16) -> (16, 16) part of network
im_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'im_mod_6{}'.format(alphabet[i])
    new_module = \
        InfConvMergeModuleIMS(
            td_chans=(ngf * 2),
            bu_chans=(ngf * 2),
            im_chans=(ngf * 2),
            rand_chans=nz1,
            conv_chans=(ngf * 2),
            use_conv=True,
            use_td_cond=use_td_cond,
            apply_bn=use_bn,
            mod_type=inf_mt,
            act_func=act_func,
            mod_name=mod_name)
    im_modules_16x16.append(new_module)

im_modules = [im_module_1] + \
             im_modules_4x4 + \
             [im_module_3] + \
             im_modules_8x8 + \
             [im_module_5] + \
             im_modules_16x16

#
# Setup a description for where to get conditional distributions from.
#
merge_info = {
    'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                 'bu_source': 'bu_mod_1', 'im_source': None},

    'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                 'bu_source': None, 'im_source': im_modules_4x4[-1].mod_name},

    'td_mod_5': {'td_type': 'pass', 'im_module': 'im_mod_5',
                 'bu_source': None, 'im_source': im_modules_8x8[-1].mod_name},

    'td_mod_7': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None},
    'td_mod_8': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None}
}

# add merge_info entries for the modules with latent variables
for i in range(depth_4x4):
    td_type = 'cond'
    td_mod_name = 'td_mod_2{}'.format(alphabet[i])
    im_mod_name = 'im_mod_2{}'.format(alphabet[i])
    im_src_name = 'im_mod_1'
    bu_src_name = 'bu_mod_3'
    if i > 0:
        im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
    if i < (depth_4x4 - 1):
        bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }
for i in range(depth_8x8):
    td_type = 'cond'
    td_mod_name = 'td_mod_4{}'.format(alphabet[i])
    im_mod_name = 'im_mod_4{}'.format(alphabet[i])
    im_src_name = 'im_mod_3'
    bu_src_name = 'bu_mod_5'
    if i > 0:
        im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
    if i < (depth_8x8 - 1):
        bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }
for i in range(depth_16x16):
    td_type = 'cond'
    td_mod_name = 'td_mod_6{}'.format(alphabet[i])
    im_mod_name = 'im_mod_6{}'.format(alphabet[i])
    im_src_name = 'im_mod_5'
    bu_src_name = 'bu_mod_7'
    if i > 0:
        im_src_name = 'im_mod_6{}'.format(alphabet[i - 1])
    if i < (depth_16x16 - 1):
        bu_src_name = 'bu_mod_6{}'.format(alphabet[i + 1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    sc_modules=[],
    merge_info=merge_info,
    output_transform=lambda x: x,
    use_sc=False
)

inf_gen_model.load_params(inf_gen_param_file)

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
log_var = sharedX(floatX([0.0]))
train_params = [log_var]

###########################################################
# Get parameters for whitening transform of training data #
###########################################################
from scipy_multivariate_normal import psd_pinv_decomposed_log_pdet, logpdf
print('computing Gauss params and log-det for fuzzy images')
mu, sigma = estimate_gauss_params(255. * Xtr)
U, log_pdet = psd_pinv_decomposed_log_pdet(sigma)
print('computing whitening transform for fuzzy images')
W, mu = estimate_whitening_transform((255. * Xtr), samples=10)

WU, log_pdet_W = psd_pinv_decomposed_log_pdet(W)

W = sharedX(W)
mu = sharedX(mu)


def whiten_data(X_sym, W_sym, mu_sym):
    # apply whitening transform to data in X
    mu_sym = T.repeat(mu_sym, X_sym.shape[0], axis=0)
    Xw_sym = X_sym - mu_sym
    Xw_sym = T.dot(Xw_sym, W_sym.T)
    return Xw_sym


######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

# whiten input
Xg_whitened = whiten_data(T.flatten(Xg, 2), W, mu)
Xg_whitened = Xg_whitened.reshape((Xg_whitened.shape[0], nc, npx, npx)).dimshuffle(0, 1, 2, 3)

######################
# Compute IWAE bound #
######################
# run an inference and reconstruction pass through the generative stuff
batch_size = Xg.shape[0]
Xg_rep = T.extra_ops.repeat(Xg_whitened, iwae_samples, axis=0)
im_res_dict = inf_gen_model.apply_im(Xg_rep)
Xg_rep_recon = im_res_dict['td_output']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

log_p_x = T.sum(log_prob_gaussian(
                T.flatten(Xg_rep, 2), T.flatten(Xg_rep_recon, 2),
                log_vars=log_var[0], do_sum=False), axis=1) + log_pdet_W


# compute quantities used in the IWAE bound
log_ws_vec = log_p_x + log_p_z - log_q_z
log_ws_mat = log_ws_vec.reshape((batch_size, iwae_samples))
ws_mat = log_ws_mat - T.max(log_ws_mat, axis=1, keepdims=True)
ws_mat = T.exp(ws_mat)
nis_weights = ws_mat / T.sum(ws_mat, axis=1, keepdims=True)
nis_weights = theano.gradient.disconnected_grad(nis_weights)

iwae_obs_costs = -1.0 * (T.sum((nis_weights * log_ws_mat), axis=1) -
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
vae_obs_klds = sum([T.mean(mod_kld.reshape((batch_size, iwae_samples)), axis=1)
                    for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

vae_bound = vae_nll_cost + vae_kld_cost

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

lrt = sharedX(0.0003)
b1t = sharedX(0.8)
train_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=1000.0)
train_updates = train_updater(train_params, iwae_bound_lme, return_grads=False)

# build training cost and update functions
t = time()
print("Compiling cost computing functions...")
# collect costs for generator parameters
g_basic_costs = [iwae_bound, vae_bound, vae_nll_cost, vae_kld_cost,
                 iwae_bound_lme]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['iwae_bound', 'vae_bound', 'vae_nll_cost', 'vae_kld_cost',
              'iwae_bound_lme']
# compile function for computing generator costs and updates
iwae_cost_func = theano.function([Xg], [log_p_x, log_p_z, log_q_z])
iwae_train_func = theano.function([Xg], [iwae_bound_lme], updates=train_updates)
g_eval_func = theano.function([Xg], g_basic_costs)
sample_func = theano.function([Z0], Xd_model)
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/EVAL.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

Xva_blocks = np.split(Xva, 5, axis=0)
for epoch in range(5):
    epoch_vae_cost = 0.0
    epoch_iwae_cost = 0.0
    for block_num, Xva_block in enumerate(Xva_blocks):
        Xva_block = shuffle(Xva_block)
        obs_count = Xva_block.shape[0]
        g_epoch_costs = [0. for c in g_basic_costs]
        g_batch_count = 0.
        for imb in tqdm(iter_data(Xva_block, size=nbatch), total=(obs_count / nbatch)):
            # transform validation batch to "image format"
            imb_img = train_transform(imb)
            # evaluate costs
            g_result = g_eval_func(imb_img)
            # train the logvar parameter
            t_result = iwae_train_func(imb_img)
            # evaluate costs more thoroughly
            iwae_bounds = iwae_multi_eval(imb_img, 10,
                                          cost_func=iwae_cost_func,
                                          iwae_num=iwae_samples)
            g_result[4] = np.mean(iwae_bounds)  # swap in tighter bound
            # accumulate costs
            g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
            g_batch_count += 1
        ##################################
        # QUANTITATIVE DIAGNOSTICS STUFF #
        ##################################
        g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
        str1 = "Epoch {}, block {}:".format(epoch, block_num)
        g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                     for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
        str2 = "    {}".format(" ".join(g_bc_strs))
        joint_str = "\n".join([str1, str2])
        print(joint_str)
        out_file.write(joint_str + "\n")
        out_file.flush()
        epoch_vae_cost += g_epoch_costs[1]
        epoch_iwae_cost += g_epoch_costs[4]
    epoch_vae_cost = epoch_vae_cost / len(Xva_blocks)
    epoch_iwae_cost = epoch_iwae_cost / len(Xva_blocks)
    epoch_bpp = nats2bpp(epoch_iwae_cost)
    str1 = "EPOCH {0:d} -- vae: {1:.2f}, iwae: {2:.2f}, bpp: {3:.2f}".format(
        epoch, epoch_vae_cost, epoch_iwae_cost, epoch_bpp)
    print(str1)
    out_file.write(str1 + "\n")
    out_file.flush()





##############
# EYE BUFFER #
##############
