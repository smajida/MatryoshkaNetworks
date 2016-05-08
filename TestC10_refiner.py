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
Xtr, Ytr, Xva, Yva = load_cifar10(data_path, va_split=5000, dtype='float32',
                                  grayscale=False)


set_seed(123)       # seed for shared rngs
nc = 3              # # of channels in image
nbatch = 100        # # of examples in batch
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
inf_mt = 0
use_td_cond = False
depth_4x4 = 1
depth_8x8 = 1
depth_16x16 = 1
refine_steps = 1

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


def check_gauss_bpp(x, x_te):
    from scipy import stats
    print('Estimating data ll with Gaussian...')
    mu, sigma = estimate_gauss_params(x, samples=10)

    # evaluate on "train" set
    x_f = fuzz_data(x, scale=1., rand_type='uniform')
    ll = stats.multivariate_normal.logpdf(x_f, (0. * mu.ravel()), sigma)
    mean_nll = -1. * np.mean(ll)
    print('  -- train gauss nll: {0:.2f}, gauss bpp: {1:.2f}'.format(mean_nll, nats2bpp(mean_nll)))

    # evaluate on "test" set
    x_f = fuzz_data(x_te, scale=1., rand_type='uniform')
    ll = stats.multivariate_normal.logpdf(x_f, (0. * mu.ravel()), sigma)
    mean_nll = -1. * np.mean(ll)
    print('  -- test gauss nll: {0:.2f}, gauss bpp: {1:.2f}'.format(mean_nll, nats2bpp(mean_nll)))

    # test with shrinking error
    alphas = [0.50, 0.25, 0.10, 0.05, 0.02]
    for alpha in alphas:
        x_f = fuzz_data(x_te, scale=1., rand_type='uniform')
        x_f = alpha * x_f
        # test with shrinking covariance
        for beta in [0.6, 0.4, 0.2, 0.1]:
            ll = stats.multivariate_normal.logpdf(x_f, (0. * mu.ravel()), (beta * sigma))
            mean_nll = -1. * np.mean(ll)
            print('  -- test a={0:.2f}, b={1:.2f}, gauss nll: {2:.2f}, gauss bpp: {3:.2f}'.format(alpha, beta, mean_nll, nats2bpp(mean_nll)))
    return

# check_gauss_bpp((255. * Xtr), (255. * Xva))

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
    out_shape=(ngf * 2, 4, 4),
    fc_dim=ngfc,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_1'
)

# grow the (4, 4) -> (4, 4) part of network
td_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'td_mod_2{}'.format(alphabet[i])
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
        mod_name=mod_name
    )
    td_modules_4x4.append(new_module)
# manual stuff for parameter sharing....

# (4, 4) -> (8, 8)
td_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='td_mod_3'
)

# grow the (8, 8) -> (8, 8) part of network
td_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'td_mod_4{}'.format(alphabet[i])
    new_module = \
    GenConvPertModule(
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
        mod_name=mod_name
    )
    td_modules_8x8.append(new_module)
# manual stuff for parameter sharing....

# (8, 8) -> (16, 16)
td_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='td_mod_5'
)

# grow the (16, 16) -> (16, 16) part of network
td_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'td_mod_6{}'.format(alphabet[i])
    new_module = \
    GenConvPertModule(
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
        mod_name=mod_name
    )
    td_modules_16x16.append(new_module)
# manual stuff for parameter sharing....

# (16, 16) -> (32, 32)
td_module_7 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='td_mod_7'
)

# (32, 32) -> (32, 32)
td_module_8 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
    apply_bn=False,
    rescale_output=True,
    use_noise=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_8'
)

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
    bu_chans=(ngf*2*4*4),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
)

# grow the (4, 4) -> (4, 4) part of network
bu_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'bu_mod_2{}'.format(alphabet[i])
    new_module = \
    BasicConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        filt_shape=(3,3),
        use_conv=use_conv,
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name=mod_name
    )
    bu_modules_4x4.append(new_module)
bu_modules_4x4.reverse() # reverse, to match "evaluation order"

# (8, 8) -> (4, 4)
bu_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
)

# grow the (8, 8) -> (8, 8) part of network
bu_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'bu_mod_4{}'.format(alphabet[i])
    new_module = \
    BasicConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        filt_shape=(3,3),
        use_conv=use_conv,
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name=mod_name
    )
    bu_modules_8x8.append(new_module)
bu_modules_8x8.reverse() # reverse, to match "evaluation order"

# (8, 8) -> (16, 16)
bu_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_5'
)

# grow the (16, 16) -> (16, 16) part of network
bu_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'bu_mod_6{}'.format(alphabet[i])
    new_module = \
    BasicConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        filt_shape=(3,3),
        use_conv=use_conv,
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name=mod_name
    )
    bu_modules_16x16.append(new_module)
bu_modules_16x16.reverse() # reverse, to match "evaluation order"

# (16, 16) -> (32, 32)
bu_module_7 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_7'
)

# (32, 32) -> (32, 32)
bu_module_8 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_8'
)

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
    out_shape=(ngf*2, 4, 4),
    fc_dim=ngfc,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_1'
)

# grow the (4, 4) -> (4, 4) part of network
im_modules_4x4 = []
for i in range(depth_4x4):
    mod_name = 'im_mod_2{}'.format(alphabet[i])
    new_module = \
    InfConvMergeModuleIMS(
        td_chans=(ngf*2),
        bu_chans=(ngf*2),
        im_chans=(ngf*2),
        rand_chans=nz1,
        conv_chans=(ngf*2),
        use_conv=True,
        use_td_cond=use_td_cond,
        apply_bn=use_bn,
        mod_type=inf_mt,
        act_func=act_func,
        mod_name=mod_name
    )
    im_modules_4x4.append(new_module)

# (4, 4) -> (8, 8)
im_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='im_mod_3'
)

# grow the (8, 8) -> (8, 8) part of network
im_modules_8x8 = []
for i in range(depth_8x8):
    mod_name = 'im_mod_4{}'.format(alphabet[i])
    new_module = \
    InfConvMergeModuleIMS(
        td_chans=(ngf*2),
        bu_chans=(ngf*2),
        im_chans=(ngf*2),
        rand_chans=nz1,
        conv_chans=(ngf*2),
        use_conv=True,
        use_td_cond=use_td_cond,
        apply_bn=use_bn,
        mod_type=inf_mt,
        act_func=act_func,
        mod_name=mod_name
    )
    im_modules_8x8.append(new_module)

# (8, 8) -> (16, 16)
im_module_5 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='im_mod_5'
)

# grow the (16, 16) -> (16, 16) part of network
im_modules_16x16 = []
for i in range(depth_16x16):
    mod_name = 'im_mod_6{}'.format(alphabet[i])
    new_module = \
    InfConvMergeModuleIMS(
        td_chans=(ngf*2),
        bu_chans=(ngf*2),
        im_chans=(ngf*2),
        rand_chans=nz1,
        conv_chans=(ngf*2),
        use_conv=True,
        use_td_cond=use_td_cond,
        apply_bn=use_bn,
        mod_type=inf_mt,
        act_func=act_func,
        mod_name=mod_name
    )
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
        im_src_name = 'im_mod_2{}'.format(alphabet[i-1])
    if i < (depth_4x4 - 1):
        bu_src_name = 'bu_mod_2{}'.format(alphabet[i+1])
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
        im_src_name = 'im_mod_4{}'.format(alphabet[i-1])
    if i < (depth_8x8 - 1):
        bu_src_name = 'bu_mod_4{}'.format(alphabet[i+1])
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
        im_src_name = 'im_mod_6{}'.format(alphabet[i-1])
    if i < (depth_16x16 - 1):
        bu_src_name = 'bu_mod_6{}'.format(alphabet[i+1])
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

############################
# Build the refiner model. #
############################

# TD modules for the refiner
td_modules_refine = []
for i in range(refine_steps):
    rtd_td_i = \
        GenConvModuleNEW(
            in_chans=nc,
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            rand_chans=nz1,
            filt_shape=(3, 3),
            act_func='lrelu',
            mod_name='rtd_td_{}'.format(i))
    rtd_cm_i = \
        BasicConvModuleNEW(
            in_chans=(ngf * 2),
            out_chans=(nc + 1),
            filt_shape=(3, 3),
            stride='single',
            act_func='ident',
            mod_name='rtd_cm_{}'.format(i))
    td_mod_refine_i = \
        TDModuleWrapperNEW(
            gen_module=rtd_td_i,
            mlp_modules=[rtd_cm_i],
            mod_name='td_mod_refine_{}'.format(i))
    td_modules_refine.append(td_mod_refine_i)

# IM modules for the refiner
im_modules_refine = []
for i in range(refine_steps):
    rim_im_i = \
        InfConvMergeModuleNEW(
            td_chans=nc,
            bu_chans=(ngf * 2),
            rand_chans=nz1,
            conv_chans=(ngf * 2),
            act_func='lrelu',
            use_td_cond=False,
            mod_name='rim_im_{}'.format(i))
    rim_cm_i = \
        BasicConvModuleNEW(
            in_chans=(nc * 3),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            stride='single',
            act_func='lrelu',
            mod_name='rim_cm_{}'.format(i))
    im_mod_refine_i = \
        IMModuleWrapperNEW(
            inf_module=rim_im_i,
            mlp_modules=[rim_cm_i],
            mod_name='im_mod_refine_{}'.format(i))
    im_modules_refine.append(im_mod_refine_i)

# BUILD THE REFINER
refiner_model = \
    DeepRefiner(
        td_modules=td_modules_refine,
        im_modules=im_modules_refine,
        ndim=4)

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
log_var = sharedX(floatX([1.0]))
noise = sharedX(floatX([noise_std]))
gen_params = inf_gen_model.gen_params + refiner_model.gen_params + [log_var]
inf_params = inf_gen_model.inf_params + refiner_model.inf_params
g_params = gen_params + inf_params

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

# quick test of log-likelihood for a basic Gaussian model...

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

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in g_params])

# run an inference and reconstruction pass through the generative stuff
im_res_dict = inf_gen_model.apply_im(Xg_whitened, noise=noise)
xg_raw = im_res_dict['td_output']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

# run an inference and reconstruction pass through the refiner
refine_dict = \
    refiner_model.apply_im(
        input_gen=xg_raw,
        input_inf=Xg_whitened,
        obs_transform=lambda x: x)
kld_dict_r = refine_dict['kld_dict']
Xg_recon = refine_dict['output']

# compute reconstruction error
log_p_x = T.sum(log_prob_gaussian(
                T.flatten(Xg_whitened, 2), T.flatten(Xg_recon, 2),
                log_vars=log_var[0], do_sum=False), axis=1)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls) - log_pdet_W

# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dict.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]

# get KL-divergences from refiner
kld_tuples_r = [(mod_name, mod_kld) for mod_name, mod_kld in kld_dict_r.items()]
vae_layer_klds_r = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples_r])
vae_layer_names_r = [mod_name for mod_name, mod_kld in kld_tuples_r]

# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples]) + sum([mod_kld for mod_name, mod_kld in kld_tuples_r])
vae_kld_cost = T.mean(vae_obs_klds)

# compute per-layer KL-divergence part of cost
alt_layer_klds = [T.sum(mod_kld**2.0, axis=1) for mod_name, mod_kld in kld_dict.items()]
alt_kld_cost = T.mean(sum(alt_layer_klds))

# compute the KLd cost to use for optimization
opt_kld_cost = (lam_kld[0] * vae_kld_cost) + ((1.0 - lam_kld[0]) * alt_kld_cost)

# combined cost for generator stuff
vae_cost = vae_nll_cost + vae_kld_cost
vae_obs_costs = vae_obs_nlls + vae_obs_klds
# cost used by the optimizer
full_cost_gen = vae_nll_cost + opt_kld_cost + vae_reg_cost
full_cost_inf = full_cost_gen

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

# stuff for performing updates
lrt = sharedX(0.0005)
b1t = sharedX(0.8)
gen_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=1000.0)
inf_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=1000.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
gen_updates, gen_grads = gen_updater(gen_params, full_cost_gen, return_grads=True)
inf_updates, inf_grads = inf_updater(inf_params, full_cost_inf, return_grads=True)
g_updates = gen_updates + inf_updates
gen_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in gen_grads]))
inf_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in inf_grads]))
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr[0:100, :]))
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost_gen, full_cost_inf, vae_cost, vae_nll_cost,
                 vae_kld_cost, gen_grad_norm, inf_grad_norm,
                 vae_obs_costs, vae_layer_klds, vae_layer_klds_r]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost_gen', 'full_cost_inf', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'gen_grad_norm', 'inf_grad_norm',
              'vae_obs_costs', 'vae_layer_klds', 'vae_layer_klds_r']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg], g_cost_outputs, updates=g_updates)    # train inference and generator
i_train_func = theano.function([Xg], g_cost_outputs, updates=inf_updates)  # train inference only
g_eval_func = theano.function([Xg], g_cost_outputs)                        # evaluate model costs
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))


n_check = 0
n_updates = 0
t = time()
kld_weights = np.linspace(0.02, 1.0, 50)
sample_z0mb = rand_gen(size=(200, nz0))
for epoch in range(1, (niter + niter_decay + 1)):
    Xtr = shuffle(Xtr)
    Xva = shuffle(Xva)
    # mess with the KLd cost
    if ((epoch - 1) < len(kld_weights)):
        lam_kld.set_value(floatX([kld_weights[epoch - 1]]))
    # lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    i_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    epoch_layer_klds_r = [0. for i in range(len(vae_layer_names_r))]
    gen_grad_norms = []
    inf_grad_norms = []
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    i_batch_count = 0.
    v_batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=(ntrain / nbatch)):
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count) * nbatch
            vmb = Xva[start_idx:(start_idx + nbatch), :]
        else:
            vmb = Xva[0:nbatch, :]
        # transform noisy training batch and carry buffer to "image format"
        imb_img = train_transform(imb)
        vmb_img = train_transform(vmb)
        # train vae on training batch
        noise.set_value(floatX([noise_std]))
        g_result = g_train_func(floatX(imb_img))
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        gen_grad_norms.append(1. * g_result[5])
        inf_grad_norms.append(1. * g_result[6])
        batch_obs_costs = g_result[7]
        batch_layer_klds = g_result[8]
        batch_layer_klds_r = g_result[9]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        epoch_layer_klds_r = [(v1 + v2) for v1, v2 in zip(batch_layer_klds_r, epoch_layer_klds_r)]
        g_batch_count += 1
        # train inference model on samples from the generator
        # if epoch > 5:
        #     smb_img = binarize_data(sample_func(rand_gen(size=(100, nz0))))
        #     i_result = i_train_func(smb_img)
        #     i_epoch_costs = [(v1 + v2) for v1, v2 in zip(i_result[:5], i_epoch_costs)]
        i_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            noise.set_value(floatX([0.0]))
            v_result = g_eval_func(vmb_img)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:6], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 15) or (epoch == 50) or (epoch == 100):
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
    gen_grad_norms = np.asarray(gen_grad_norms)
    inf_grad_norms = np.asarray(inf_grad_norms)
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    i_epoch_costs = [(c / i_batch_count) for c in i_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    epoch_layer_klds_r = [(c / g_batch_count) for c in epoch_layer_klds_r]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    i_bc_strs = ["{0:s}: {1:.2f},".format(c_name, i_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2i = " ".join(i_bc_strs)
    ggn_qtiles = np.percentile(gen_grad_norms, [50., 80., 90., 95.])
    str3 = "    [q50, q80, q90, q95, max](ggn): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        ggn_qtiles[0], ggn_qtiles[1], ggn_qtiles[2], ggn_qtiles[3], np.max(gen_grad_norms))
    ign_qtiles = np.percentile(inf_grad_norms, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](ign): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        ign_qtiles[0], ign_qtiles[1], ign_qtiles[2], ign_qtiles[3], np.max(inf_grad_norms))
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str6 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str7 = "    module kld -- {}".format(" ".join(kld_strs))
    kld_strs_r = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names_r, epoch_layer_klds_r)]
    str7_r = "    refine kld -- {}".format(" ".join(kld_strs_r))
    str8 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, bpp: {2:.2f}, vfe/iwae: {3:.2f}".format(
        v_epoch_costs[3], v_epoch_costs[4], nats2bpp(v_epoch_costs[2]), v_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str2i, str3, str4, str5, str6, str7, str7_r, str8])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    if (epoch < 20) or (((epoch - 1) % 20) == 0):
        # generate some samples from the model prior
        samples = np.asarray(sample_func(sample_z0mb))
        color_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(result_dir, epoch))
        # test reconstruction performance (inference + generation)
        tr_rb = Xtr[0:100, :]
        va_rb = Xva[0:100, :]
        # get the model reconstructions
        tr_rb = train_transform(tr_rb)
        va_rb = train_transform(va_rb)
        tr_recons = recon_func(tr_rb)
        va_recons = recon_func(va_rb)
        # stripe data for nice display (each reconstruction next to its target)
        tr_vis_batch = np.zeros((200, nc, npx, npx))
        va_vis_batch = np.zeros((200, nc, npx, npx))
        for rec_pair in range(100):
            idx_in = 2 * rec_pair
            idx_out = 2 * rec_pair + 1
            tr_vis_batch[idx_in, :, :, :] = tr_rb[rec_pair, :, :, :]
            tr_vis_batch[idx_out, :, :, :] = tr_recons[rec_pair, :, :, :]
            va_vis_batch[idx_in, :, :, :] = va_rb[rec_pair, :, :, :]
            va_vis_batch[idx_out, :, :, :] = va_recons[rec_pair, :, :, :]
        # draw images...
        color_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(result_dir, epoch))
        color_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(result_dir, epoch))







##############
# EYE BUFFER #
##############
