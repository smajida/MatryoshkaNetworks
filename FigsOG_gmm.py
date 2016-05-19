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


def transpose_images(img_ary, grid_shape):
    '''
    Transpose the grid location of the images in the rows of img_ary.
    '''
    rows = grid_shape[0]
    cols = grid_shape[1]
    img_ary_T = np.zeros(img_ary.shape)
    i = 0
    for col in range(cols):
        for row in range(rows):
            j = (row * cols) + col
            img_ary_T[i, :] = img_ary[j, :]
            i += 1
    return img_ary_T


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

# get shapes for all latent variables in model
# -- shape is None for TD modules that don't use z
z_shapes = []
for tdm in td_modules:
    if hasattr(tdm, 'rand_shape'):
        z_shape = [d for d in tdm.rand_shape]
        if len(z_shape) > 1:
            # correct shape for carrying nz0 forward
            z_shape[0] = z_shape[0] - nz0
        z_shapes.append(tuple(z_shape))
    else:
        z_shapes.append(None)
# collect list of TD modules that actually use z
td_z_modules = [tdm for tdm in td_modules if hasattr(tdm, 'rand_shape')]

print('z_shapes: {}'.format(z_shapes))


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output

###############################################
# BUILD FUNCTIONS FOR SAMPLING FROM THE MODEL #
###############################################

# setup symbolic vars for the model
x_in = T.tensor4()
z_in = []
for z_shape in z_shapes:
    if z_shape is not None:
        if len(z_shape) == 1:
            z_in.append(T.matrix())
        else:
            z_in.append(T.tensor4())
    else:
        z_in.append(None)
z_rand = [z for z in z_in if z is not None]

# run an inference pass through to get info about posterior distributions
im_res_dict = inf_gen_model.apply_im(x_in, kl_mode='analytical')
x_recon = clip_sigmoid(im_res_dict['td_output'])
z_samps = [im_res_dict['z_dict'][tdm.mod_name] for tdm in td_z_modules]
mix_comp_post = im_res_dict['mix_comp_post']

# run an un-grounded pass through generative stuff for sampling from model
x_from_z = inf_gen_model.apply_td(rand_vals=z_in)
x_from_z = clip_sigmoid(x_from_z)


###############################################
# BUILD THEANO FUNCTIONS AND HELPER FUNCTIONS #
###############################################
t = time()
print("Compiling sampling and reconstruction functions...")
# make function to collect posterior latent samples and posterior mixture weights
mix_post_func = theano.function([x_in], mix_comp_post)
post_sample_func = theano.function([x_in], z_samps)
# make function to sample from model given all the latent vars
sample_func = theano.function(z_rand, x_from_z)
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
    x_samples = sample_func(*z_all_scale)
    return x_samples


def complete_z_samples(z_samps_partial, z_modules):
    '''
    Complete the given set of latent samples, to match the shapes required by
    the main model (i.e. inf_gen_model).
    '''
    assert (len(z_samps_partial) >= 1)
    obs_count = z_samps_partial[0].shape[0]
    z_samps_full = [z for z in z_samps_partial]
    for i, tdm in enumerate(z_modules):
        if i >= len(z_samps_partial):
            z_shape = [obs_count] + [d for d in tdm.rand_shape]
            z_shape[1] = z_shape[1] - nz0
            # z_shape[1] = z_shape[1]
            z_samps = rand_gen(size=tuple(z_shape))
            z_samps_full.append(z_samps)
    return z_samps_full


##############################################################
# Draw random samples conditioned on each mixture component. #
##############################################################

# test posterior info function
x_in = train_transform(Xva[:100, :])
mix_weights = mix_post_func(x_in)
post_samples = post_sample_func(x_in)

# sample at random from the mixture components
comp_idx = range(mix_comps)
z_mix = mix_module.sample_mix_comps(comp_idx=comp_idx, batch_size=None)
z_mix = np.repeat(z_mix, 10, axis=0)
z_rand = complete_z_samples([z_mix], td_z_modules)
x_samples = sample_func_scaled(z_rand, 0.9, no_scale=[0])


print('Collecting posterior info for validation set...')
batch_size = 200
va_batches = []
va_post_samples = [[] for z in z_rand]
va_mix_posts = []
for i in range(20):
    b_start = i * batch_size
    b_end = b_start + batch_size
    xmb = train_transform(Xva[b_start:b_end, :])
    va_batches.append(xmb)
    # collect samples from approximate posteriors for xmb
    post_z = post_sample_func(xmb)
    for zs_list, zs in zip(va_post_samples, post_z):
        zs_list.append(zs)
    # compute posterior mixture weights for xmb
    va_mix_posts.append(mix_post_func(xmb))
# group up output of the batch computations
va_batches = np.concatenate(va_batches, axis=0)
va_post_samples = [np.concatenate(ary_list, axis=0) for ary_list in va_post_samples]
va_mix_posts = np.concatenate(va_mix_posts, axis=0)
for i, vaps in enumerate(va_post_samples):
    print('va_post_samples[{}].shape: {}'.format(i, vaps.shape))
print('va_mix_posts.shape: {}'.format(va_mix_posts.shape))

# collect indices of examples sorted most strongly into each mixture component
# -- indices are indices into va_post_samples and va_mix_posts
obs_count = va_mix_posts.shape[0]
mix_comp_members = []
comp_assignments = np.argmax(va_mix_posts, axis=1)
for comp_idx in range(mix_comps):
    # get examples assigned to this component, after sorting based
    # on strength of assignment to the component
    comp_members = np.asarray([i for i in range(obs_count)
                               if comp_assignments[i] == comp_idx])
    # get posterior mixture weights for members of this component
    comp_weights = va_mix_posts[comp_members, comp_idx]
    sort_idx = np.argsort(-comp_weights)  # sort greatest -> least
    comp_members = comp_members[sort_idx]  # sorted list of members
    mix_comp_members.append(comp_members)

# grab a few members from each component
comp_reps = 4
fig_members = []
for comp_idx in range(mix_comps):
    fig_members.append(mix_comp_members[comp_idx][:comp_reps])
fig_members = np.concatenate(fig_members, axis=0)

# # collect data and posterior samples for each mixture component
# mix_data_samples = []
# for comp_idx in range(mix_comps):
#     comp_members = mix_comp_members[comp_idx]
#     comp_data_samples = va_batches[comp_members, :]
#     mix_data_samples.append(comp_data_samples)

# # draw examples from each mixture component
# comp_reps = 4
# comp_count = mix_comps
# mix_examples = np.zeros((comp_count * comp_reps, nc, npx, npx))
# mix_post_samples = []
# for i in range(comp_count):
#     s_idx = i * comp_reps
#     e_idx = s_idx + comp_reps
#     mix_examples[s_idx:e_idx, :, :, :] = mix_data_samples[i][:comp_reps, :, :, :]
# real_samples = draw_transform(mix_examples)
# grayscale_grid_vis(real_samples, (1, comp_reps * comp_count),
#                    "{}/fig_mix_examples.png".format(result_dir))

# # generate from samples from the mixture components.
# # -- each round of sampling produces (comp_count * comp_reps) samples...
# fix_depths = [0, 1, 3, 5, 7]
# fix_depth_samples = [real_samples]
# comp_idx = np.arange(comp_count)
# for fd in fix_depths:
#     lvar_samps = []
#     # generate the "fixed" latent variables
#     for j in range(len(z_shapes)):
#         if (j == 0) and (fd == 0):
#             z_mix = mix_module.sample_mix_comps(comp_idx=comp_idx.repeat(comp_reps), batch_size=None)
#             lvar_samps.append(z_mix)
#         elif (j == 0) and (fd > 0):
#             z_mix = mix_module.sample_mix_comps(comp_idx=comp_idx, batch_size=None)
#             z_mix = np.repeat(z_mix, comp_reps, axis=0)
#             lvar_samps.append(z_mix)
#         elif z_shapes[j] is not None:
#             z_shape = [d for d in z_shapes[j]]
#             if j < fd:
#                 samp_shape = [comp_count] + z_shape
#                 z_samps = rand_gen(size=tuple(samp_shape))
#                 z_samps = np.repeat(z_samps, comp_reps, axis=0)
#             else:
#                 samp_shape = [comp_count * comp_reps] + z_shape
#                 z_samps = rand_gen(size=tuple(samp_shape))
#             lvar_samps.append(z_samps)
#     # sample using the generated latent variables
#     samples = sample_func_scaled(lvar_samps, 1.0, no_scale=[0])
#     samples = draw_transform(samples)
#     fix_depth_samples.append(samples)
# # stack the samples from each "fix depth"
# samples = draw_transform(np.vstack(fix_depth_samples))
# print('samples.shape: {}'.format(samples.shape))
# grayscale_grid_vis(samples, (len(fix_depth_samples), comp_count * comp_reps),
#                    "{}/fig_mix_samples.png".format(result_dir))








##############
# EYE BUFFER #
##############
