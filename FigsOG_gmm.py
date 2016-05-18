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
from lib.ops import log_mean_exp, binarize_data
from lib.costs import log_prob_bernoulli
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import load_omniglot

#
# Phil's business
#
from ModelBuilders import build_og_conv_res, build_og_conv_res_hires

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./omniglot"

# setup paths for dumping diagnostic info
desc = 'test_conv_5deep_lores_gmm_mc50_pe04'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load MNIST dataset, either fixed or dynamic binarization
data_path = "{}/data/".format(EXP_DIR)
Xtr, Ytr, Xva, Yva = load_omniglot(data_path, target_type='one-hot')


set_seed(123)        # seed for shared rngs
nbatch = 100         # # of examples in batch
nc = 1               # # of channels in image
nz0 = 32             # # of dim in top-most latent variables
nz1 = 6              # # of dim in intermediate latent variables
ngf = 32             # base # of filters for conv layers
ngfc = 256           # # of dim in top-most hidden layer
npx = 28             # # of pixels width/height of images
nx = npx * npx * nc  # # of dimensions in X
niter = 200          # # of iter at starting learning rate
niter_decay = 200    # # of iter to linearly decay learning rate to zero
use_td_cond = False
depth_7x7 = 5
depth_14x14 = 5
depth_28x28 = None
mix_comps = 50

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xtr.shape[0]


def train_transform(X):
    # transform vectorized observations into convnet inputs
    X = binarize_data(X)
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

# BUILD THE MODEL
if depth_28x28 is None:
    inf_gen_model = \
        build_og_conv_res(
            nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc, mix_comps=mix_comps,
            use_bn=False, act_func='lrelu', use_td_cond=use_td_cond,
            depth_7x7=depth_7x7, depth_14x14=depth_14x14)
else:
    inf_gen_model = \
        build_og_conv_res_hires(
            nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc, mix_comps=mix_comps,
            use_bn=False, act_func='lrelu', use_td_cond=use_td_cond,
            depth_7x7=depth_7x7, depth_14x14=depth_14x14, depth_28x28=depth_28x28)
td_modules = inf_gen_model.td_modules
bu_modules = inf_gen_model.bu_modules
im_modules = inf_gen_model.im_modules
mix_module = inf_gen_model.mix_module

# load pretrained model parameters
inf_gen_model.load_params(inf_gen_param_file)


# get an ordered list of TD modules that require latent samples
td_z_modules = [tdm for tdm in td_modules if
                (inf_gen_model.merge_info[tdm.mod_name]['td_type'] in ['top', 'cond'])]
z_shapes = [tdm.rand_shape for tdm in td_z_modules]


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output

###############################################
# BUILD FUNCTIONS FOR SAMPLING FROM THE MODEL #
###############################################

# Setup symbolic vars for the model inputs, outputs, and costs
X_in = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Z_all = [T.matrix()] + [T.tensor4() for tdm in td_z_modules[1:]]  # symbolic vars for each set of latent variables

# run an inference pass through to get info about posterior distributions
im_res_dict = inf_gen_model.apply_im(X_in, kl_mode='analytical')
X_recon = clip_sigmoid(im_res_dict['td_output'])
z_samps = im_res_dict['z_dict']
mix_comp_weight = im_res_dict['mix_comp_weight']

# run an un-grounded pass through generative stuff for sampling from model
X_from_Z = inf_gen_model.apply_td(rand_vals=Z_all)
X_from_Z = clip_sigmoid(X_from_Z)


###############################################
# BUILD THEANO FUNCTIONS AND HELPER FUNCTIONS #
###############################################

print("Compiling sampling and reconstruction functions...")
# make function to collect posterior latent samples and posterior mixture weights
info_func = theano.function([X_in], [X_recon, z_samps, mix_comp_weight])
# make function to sample from model given all the latent vars
sample_func = theano.function(Z_all, X_from_Z)
print "{0:.2f} seconds to compile theano functions".format(time() - t)


def sample_func_scaled(z_all, z_scale, no_scale):
    '''
    Rescale the latent variables, and sample likelier samples.
    '''
    z_all_scale = []
    for i, z in enumerate(z_all):
        if i in no_scale:
            z_all_scale.append(z)
        else:
            z_all_scale.append(z_scale * z)
    x_samples = sample_func(z_all_scale)
    return x_samples

def complete_z_samples(z_samps_partial, z_modules):
    '''
    Complete the given set of latent samples, to match the shapes required by
    the main model (i.e. inf_gen_model).
    '''
    obs_count = z_samps_partial[0].shape[0]
    z_samps_full = [z for z in z_samps_partial]
    for i, tdm in enumerate(z_modules):
        if i >= len(z_samps_partial):
            z_shape = tuple([obs_count] + [d for d in tdm.rand_shape])
            z_samps = rand_gen(size=z_shape)
            z_samps_full.append(z_samps)
    return z_samps_full


##############################################################
# Draw random samples conditioned on each mixture component. #
##############################################################

# sample at random from the mixture components
comp_idx = range(mix_comps)
z_mix = mix_module.sample_mix_comps(comp_idx=comp_idx, batch_size=None)
z_all = complete_z_samples([z_mix], td_z_modules)
x_samples = sample_func_scaled(z_all, 0.9, no_scale=[0])

# test posterior info function
x_in = train_transform(Xva[:100, :])
post_info = info_func(x_in)


# for i in range(min(6, len(z_shapes))):
#     lvar_samps = []
#     # generate the "fixed" latent variables
#     for j in range(len(z_shapes)):
#         samp_shape = [d for d in z_shapes[j]]
#         if j < i:
#             samp_shape[0] = 20
#             z_samps = rand_gen(size=tuple(samp_shape))
#             z_samps = np.repeat(z_samps, 25, axis=0)
#         else:
#             samp_shape[0] = 400
#             z_samps = rand_gen(size=tuple(samp_shape))
#         lvar_samps.append(z_samps)
#     # sample using the generated latent variables
#     samples = np.asarray(sample_func(*lvar_samps))
#     grayscale_grid_vis(draw_transform(samples), (20, 20), "{}/eval_gen_e{}_b{}_{}fix.png".format(result_dir, epoch, block_num, i))








##############
# EYE BUFFER #
##############
