import os
import json
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.costs import log_prob_gaussian, log_prob_bernoulli
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import load_svhn

#
# Phil's business
#
from MatryoshkaModules import BasicConvModule, \
                              GenConvDblResModule, GenConvResModule, \
                              GenFCModule, InfConvMergeModule, \
                              InfFCModule, BasicConvResModule, \
                              DiscConvResModule, DiscFCModule
from MatryoshkaNetworks import InfGenModel, DiscNetworkGAN, GenNetworkGAN

# path for dumping experiment info and fetching dataset
EXP_DIR = "./svhn"
DATA_SIZE = 400000

# setup paths for dumping diagnostic info
desc = 'test_van_match_dm3_drop00_more_vae'
model_dir = "{}/models/{}".format(EXP_DIR, desc)
sample_dir = "{}/samples/{}".format(EXP_DIR, desc)
log_dir = "{}/logs".format(EXP_DIR)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# locations of 32x32 SVHN dataset
tr_file = "{}/data/train_32x32.mat".format(EXP_DIR)
te_file = "{}/data/test_32x32.mat".format(EXP_DIR)
ex_file = "{}/data/extra_32x32.mat".format(EXP_DIR)
# load dataset (load more when using adequate computers...)
data_dict = load_svhn(tr_file, te_file, ex_file=ex_file, ex_count=DATA_SIZE)

# stack data into a single array and rescale it into [-1,1] (per observation)
Xtr = np.concatenate([data_dict['Xtr'], data_dict['Xex']], axis=0)
del data_dict
Xtr = Xtr - np.min(Xtr, axis=1, keepdims=True)
Xtr = Xtr / np.max(Xtr, axis=1, keepdims=True)
Xtr = 2.0 * (Xtr - 0.5)
Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
Xtr_std = np.std(Xtr, axis=0, keepdims=True)
# split into training and validation sets (for checking VAE overfitting)
Xtr = Xtr[:-10000,:]
Xva = Xtr[-10000:,:]


set_seed(1)       # seed for shared rngs
l2 = 1.0e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz0 = 64         # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngf = 64          # base # of filters for conv layers in generative stuff
ndf = 64          # base # of filters for conv layers in discriminator
ndfc = 256        # # of filters in fully connected layers of discriminator
ngfc = 256        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
niter = 75       # # of iter at starting learning rate
niter_decay = 75 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
multi_rand = True # whether to use stochastic variables at multiple scales
multi_disc = True # whether to use discriminator feedback at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_er = True     # whether to use "experience replay"
use_annealing = True # whether to anneal the target distribution while training
use_carry = True     # whether to carry difficult VAE inputs to the next batch
carry_count = 16        # number of stubborn VAE inputs to carry to next batch
er_buffer_size = 250000 # size of the "experience replay" buffer
drop_rate = 0.0
ntrain = Xtr.shape[0]


def train_transform(X):
    # transform vectorized observations into convnet inputs
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3))

def draw_transform(X):
    # transform vectorized observations into drawable images
    X = (X + 1.0) * 127.0
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1))

def rand_gen(size, noise_type='normal'):
    if noise_type == 'normal':
        r_vals = floatX(np_rng.normal(size=size))
    elif noise_type == 'uniform':
        r_vals = floatX(np_rng.uniform(size=size, low=-1.0, high=1.0))
    else:
        assert False, "unrecognized noise type!"
    return r_vals

def update_exprep_buffer(er_buffer, generator, replace_frac=0.1, do_print=False):
    """
    Update the "experience replay buffer" er_buffer using samples generated by
    generator. Replace replace_frac of buffer with new samples.

    Assume er_buffer is a 2d numpy array.
    """
    buffer_size = er_buffer.shape[0]
    new_sample_count = int(buffer_size * replace_frac)
    new_samples = floatX(np.zeros((new_sample_count, nc*npx*npx)))
    start_idx = 0
    end_idx = 500
    if do_print:
        print("Updating experience replay buffer...")
    while start_idx < new_sample_count:
        samples = generator.generate_samples(500)
        samples = samples.reshape((500,-1))
        end_idx = min(end_idx, new_sample_count)
        new_samples[start_idx:end_idx,:] = samples[:(end_idx-start_idx),:]
        start_idx += 500
        end_idx += 500
    idx = np.arange(buffer_size)
    npr.shuffle(idx)
    replace_idx = idx[:new_sample_count]
    er_buffer[replace_idx,:] = new_samples
    return er_buffer

def sample_exprep_buffer(er_buffer, sample_count):
    """
    Sample from the "experience replay buffer" er_buffer, with replacement.
    """
    buffer_size = er_buffer.shape[0]
    idx = npr.randint(0,high=buffer_size)
    samples = er_buffer[idx,:]
    return samples


def gauss_blur(x, x_std, w_x, w_g):
    """
    Add gaussian noise to x, with rescaling to keep variance constant w.r.t.
    the initial variance of x (in x_var). w_x and w_g should be weights for
    a convex combination.
    """
    g_std = np.sqrt( (x_std * (1. - w_x)**2.) / (w_g**2. + 1e-4) )
    g_noise = g_std * np_rng.normal(size=x.shape)
    x_blurred = w_x*x + w_g*g_noise
    return floatX(x_blurred)

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

#########################################
# Setup the top-down processing modules #
# -- these do generation                #
#########################################

td_module_1 = \
GenFCModule(
    rand_dim=nz0,
    out_shape=(ngf*8, 2, 2),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=True,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 2, 2)

td_module_2 = \
GenConvResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 4, 4)

td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*2, 8, 8)

td_module_4 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_4'
) # output is (batch, ngf*2, 16, 16)

td_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_5'
) # output is (batch, ngf*1, 32, 32)

td_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
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

bu_module_1 = \
InfFCModule(
    bu_chans=(ngf*8*2*2),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    act_func='lrelu',
    mod_name='bu_mod_1'
) # output is (batch, nz0), (batch, nz0)

bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*8),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='lrelu',
    mod_name='bu_mod_2'
) # output is (batch, ngf*4, 2, 2)

bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='lrelu',
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 4, 4)

bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='lrelu',
    mod_name='bu_mod_4'
) # output is (batch, ngf*2, 8, 8)

bu_module_5 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='lrelu',
    mod_name='bu_mod_5'
) # output is (batch, ngf*2, 16, 16)

bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=False,
    stride='single',
    act_func='lrelu',
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
    act_func='lrelu',
    mod_name='im_mod_2'
) # merge input to td_mod_2 and output of bu_mod_2, to place a distribution
  # over the rand_vals used in td_mod_2.

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=True,
    act_func='lrelu',
    mod_name='im_mod_3'
) # merge input to td_mod_3 and output of bu_mod_3, to place a distribution
  # over the rand_vals used in td_mod_3.

im_module_4 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*1),
    use_conv=True,
    act_func='lrelu',
    mod_name='im_mod_4'
) # merge input to td_mod_4 and output of bu_mod_4, to place a distribution
  # over the rand_vals used in td_mod_4.

im_module_5 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*1),
    use_conv=True,
    act_func='lrelu',
    mod_name='im_mod_5'
) # merge input to td_mod_5 and output of bu_mod_5, to place a distribution
  # over the rand_vals used in td_mod_5.

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
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'},
}

dist_scale = sharedX( floatX([0.1]) )
# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=tanh,
    dist_scale=dist_scale[0]
)
# create a model of just the generator
gen_network = GenNetworkGAN(modules=td_modules, output_transform=tanh)

#####################################
#####################################
## Setup the discriminator network ##
#####################################
#####################################

disc_module_1 = \
BasicConvModule(
    filt_shape=(5,5),
    in_chans=nc,
    out_chans=(ndf*1),
    apply_bn=False,
    unif_drop=drop_rate,
    chan_drop=0.0,
    stride='double',
    act_func='lrelu',
    mod_name='disc_mod_1'
) # output is (batch, ndf*1, 16, 16)

disc_module_2 = \
DiscConvResModule(
    in_chans=(ndf*1),
    out_chans=(ndf*2),
    conv_chans=(ndf*1),
    filt_shape=(5,5),
    use_conv=False,
    unif_drop=0.0,
    chan_drop=drop_rate,
    ds_stride=2,
    mod_name='disc_mod_2'
) # output is (batch, ndf*2, 8, 8)

disc_module_3 = \
DiscConvResModule(
    in_chans=(ndf*2),
    out_chans=(ndf*4),
    conv_chans=(ndf*2),
    filt_shape=(5,5),
    use_conv=False,
    unif_drop=0.0,
    chan_drop=drop_rate,
    ds_stride=2,
    mod_name='disc_mod_3'
) # output is (batch, ndf*4, 4, 4)

disc_module_4 = \
DiscConvResModule(
    in_chans=(ndf*4),
    out_chans=(ndf*8),
    conv_chans=(ndf*4),
    filt_shape=(5,5),
    use_conv=False,
    unif_drop=0.0,
    chan_drop=drop_rate,
    ds_stride=2,
    mod_name='disc_mod_4'
) # output is (batch, ndf*4, 2, 2)

disc_module_5 = \
DiscFCModule(
    fc_dim=ndfc,
    in_dim=(ndf*8*2*2),
    use_fc=False,
    apply_bn=True,
    unif_drop=drop_rate,
    mod_name='disc_mod_5'
) # output is (batch, 1)

disc_modules = [disc_module_1, disc_module_2, disc_module_3,
                disc_module_4, disc_module_5]

# Initialize the discriminator network
disc_network = DiscNetworkGAN(modules=disc_modules)
d_params = disc_network.params


####################################
# Setup the optimization objective #
####################################
lam_vae = sharedX(np.ones((1,)).astype(theano.config.floatX))
lam_kld = sharedX(np.ones((1,)).astype(theano.config.floatX))
obs_logvar = sharedX(np.zeros((1,)).astype(theano.config.floatX))
bounded_logvar = 1.0 * tanh((1.0/2.0) * obs_logvar)
gen_params = [obs_logvar] + inf_gen_model.gen_params
inf_params = [dist_scale] + inf_gen_model.inf_params
g_params = gen_params + inf_params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

if multi_disc:
    # multi-scale discriminator guidance
    ret_vals = range(1, len(disc_network.modules))
else:
    # full-scale discriminator guidance only
    ret_vals = [ (len(disc_network.modules)-1) ]

# Setup symbolic vars for the model inputs, outputs, and costs
Xd = T.tensor4()  # symbolic var for inputs to discriminator
Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Xr = T.tensor4()  # symbolic var for inputs to discriminator, from er buffer
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# run an inference and reconstruction pass through the generative stuff
Xg_recon, kld_dicts = inf_gen_model.apply_im(Xg)
# feed reconstructions and their instigators into the discriminator.
# -- these values are used for training the generative stuff ONLY.
# -- use same drop mask for all examples, to get sensible reconstruction loss.
Xg_joint = T.concatenate([Xg, Xg_recon], axis=0)
Hg_joint, _ = disc_network.apply(input=Xg_joint, ret_vals=ret_vals,
                                 ret_acts=True, app_sigm=True,
                                 share_mask=True)
Hg_flat = [T.flatten(hg,2) for hg in Hg_joint]
Hg_world = [hg[:Xg.shape[0],:] for hg in Hg_flat]
Hg_recon = [hg[Xg.shape[0]:,:] for hg in Hg_flat]

vae_layer_nlls = []
for hg_world, hg_recon in zip(Hg_world, Hg_recon):
    lnll = -1. * log_prob_gaussian(hg_world, hg_recon,
                                   log_vars=bounded_logvar[0], do_sum=False,
                                   use_huber=0.25)
    # NLLs are recorded for each observation in the batch
    vae_layer_nlls.append(T.sum(lnll, axis=1))
print("len(vae_layer_nlls): {}".format(len(vae_layer_nlls)))
#vae_obs_nlls = vae_layer_nlls[0]
#vae_obs_nlls = vae_layer_nlls[2]
vae_obs_nlls = vae_layer_nlls[3]
#vae_obs_nlls = vae_layer_nlls[4]
vae_nll_cost = T.mean(vae_obs_nlls)

# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dicts.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]
# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

# parameter regularization part of cost
vae_reg_cost = 2e-5 * sum([T.sum(p**2.0) for p in g_params])
# combined cost for generator stuff
vae_cost = vae_nll_cost + (lam_kld[0] * vae_kld_cost) + vae_reg_cost
vae_obs_costs = vae_obs_nlls + vae_obs_klds


##########################################################
# CONSTRUCT COST VARIABLES FOR THE GAN PART OF OBJECTIVE #
##########################################################
# run an un-grounded pass through generative stuff for GAN-style training
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)
# feed "world-generated" data and "model-generated" data into the discriminator
Hd_model, Yd_model = disc_network.apply(input=Xd_model, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
Hd_world, Yd_world = disc_network.apply(input=Xd, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
Hd_exrep, Yd_exrep = disc_network.apply(input=Xr, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
# compute classification parts of GAN cost (for generator and discriminator)
gan_layer_nlls_world = []
gan_layer_nlls_model = []
gan_layer_nlls_exrep = []
gan_layer_nlls_gnrtr = []
weights = [1. for yd_world in Yd_world]
scale = sum(weights)
weights = [w/scale for w in weights]
for yd_world, yd_model, yd_exrep, w in zip(Yd_world, Yd_model, Yd_exrep, weights):
    lnll_world = bce(yd_world, T.ones_like(yd_world))
    lnll_model = bce(yd_model, T.zeros_like(yd_model))
    lnll_exrep = bce(yd_exrep, T.zeros_like(yd_exrep))
    lnll_gnrtr = bce(yd_model, T.ones_like(yd_model))
    gan_layer_nlls_world.append(w * T.mean(lnll_world))
    gan_layer_nlls_model.append(w * T.mean(lnll_model))
    gan_layer_nlls_exrep.append(w * T.mean(lnll_exrep))
    gan_layer_nlls_gnrtr.append(w * T.mean(lnll_gnrtr))
gan_nll_cost_world = sum(gan_layer_nlls_world)
gan_nll_cost_model = sum(gan_layer_nlls_model)
gan_nll_cost_exrep = sum(gan_layer_nlls_exrep)
gan_nll_cost_gnrtr = sum(gan_layer_nlls_gnrtr)

# parameter regularization parts of GAN cost
gan_reg_cost_d = 2e-5 * sum([T.sum(p**2.0) for p in d_params])
gan_reg_cost_g = 2e-5 * sum([T.sum(p**2.0) for p in gen_params])
# compute GAN cost for discriminator
if use_er:
    adv_cost = (0.5 * gan_nll_cost_model) + (0.5 * gan_nll_cost_exrep)
    gan_cost_d = gan_nll_cost_world + adv_cost + gan_reg_cost_d
else:
    gan_cost_d = gan_nll_cost_world + gan_nll_cost_model + gan_reg_cost_d

# compute GAN cost for generator
gan_cost_g = gan_nll_cost_gnrtr + gan_reg_cost_g

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################
full_cost_d = gan_cost_d
full_cost_gen = gan_cost_g + (lam_vae[0] * vae_cost)
#full_cost_gen = gan_cost_g
full_cost_inf = vae_cost

# stuff for performing updates
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)
gen_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4, clipnorm=100.0)
inf_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4, clipnorm=1000.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
d_updates, d_grads = d_updater(d_params, full_cost_d, return_grads=True)
gen_updates, gen_grads = gen_updater(gen_params, full_cost_gen, return_grads=True)
inf_updates, inf_grads = inf_updater(inf_params, full_cost_inf, return_grads=True)
g_updates = gen_updates + inf_updates
gen_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in gen_grads]))
inf_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in inf_grads]))
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr[0:100,:])) # cheeky model implementation test
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost_gen, full_cost_inf, gan_cost_g, vae_cost,
                 vae_nll_cost, vae_kld_cost, gen_grad_norm, inf_grad_norm,
                 vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost_gen', 'full_cost_inf', 'gan_cost_g', 'vae_cost',
              'vae_nll_cost', 'vae_kld_cost', 'gen_grad_norm', 'inf_grad_norm',
              'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg, Z0], g_cost_outputs, updates=g_updates)
# collect costs for discriminator parameters
d_basic_costs = [full_cost_d, gan_cost_d, gan_nll_cost_world, gan_nll_cost_model]
d_bc_idx = range(0, len(d_basic_costs))
d_bc_names = ['full_cost_d', 'gan_cost_d', 'gan_nll_cost_world', 'gan_nll_cost_model']
d_cost_outputs = d_basic_costs
# compile function for computing discriminator costs and updates
if use_er:
    d_train_func = theano.function([Xd, Z0, Xr], d_cost_outputs, updates=d_updates)
else:
    d_train_func = theano.function([Xd, Z0], d_cost_outputs, updates=d_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# initialize an experience replay buffer
er_buffer = floatX(np.zeros((er_buffer_size, nc*npx*npx)))
start_idx = 0
end_idx = 1000
print("Initializing experience replay buffer...")
while start_idx < er_buffer_size:
    samples = gen_network.generate_samples(1000)
    samples = samples.reshape((1000,-1))
    end_idx = min(end_idx, er_buffer_size)
    er_buffer[start_idx:end_idx,:] = samples[:(end_idx-start_idx),:]
    start_idx += 1000
    end_idx += 1000
print("DONE.")
# initialize a buffer holding VAE inputs to carry to next batch
carry_buffer = Xtr[0:carry_count,:].copy()

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(sample_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
gauss_blur_weights = np.linspace(0.0, 1.0, 15) # weights for distribution "annealing"
sample_z0mb = rand_gen(size=(200, nz0))        # root noise for visualizing samples
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    Xva = shuffle(Xva)
    vae_scale = 0.05
    kld_scale = 1.0
    lam_vae.set_value(np.asarray([vae_scale]).astype(theano.config.floatX))
    lam_kld.set_value(np.asarray([kld_scale]).astype(theano.config.floatX))
    g_epoch_costs = [0. for i in range(6)]
    v_epoch_costs = [0. for i in range(6)]
    d_epoch_costs = [0. for i in range(4)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    gen_grad_norms = []
    inf_grad_norms = []
    carry_costs = []
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    v_batch_count = 0.
    d_batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        if epoch < gauss_blur_weights.shape[0]:
            w_x = gauss_blur_weights[epoch]
        else:
            w_x = 1.0
        w_g = 1.0 - w_x
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count)*100
            vmb = Xva[start_idx:(start_idx+100),:]
        else:
            vmb = Xva[0:100,:]
        if use_annealing and (w_x < 0.999):
            # add noise to both the current batch and the carry buffer
            imb_fuzz = np.clip(gauss_blur(imb, Xtr_std, w_x, w_g),
                               a_min=-1.0, a_max=1.0)
            vmb_fuzz = np.clip(gauss_blur(vmb, Xtr_std, w_x, w_g),
                               a_min=-1.0, a_max=1.0)
            cb_fuzz = np.clip(gauss_blur(carry_buffer, Xtr_std, w_x, w_g),
                              a_min=-1.0, a_max=1.0)
        else:
            # use noiseless versions of the current batch and the carry buffer
            imb_fuzz = imb.copy()
            vmb_fuzz = vmb.copy()
            cb_fuzz = carry_buffer.copy()
        # transform noisy training batch and carry buffer to "image format"
        imb_fuzz = train_transform(imb_fuzz)
        vmb_fuzz = train_transform(vmb_fuzz)
        cb_fuzz = train_transform(cb_fuzz)
        # sample random noise for top of generator
        z0 = rand_gen(size=(nbatch, nz0))
        # sample data from experience replay buffer
        xer = train_transform(sample_exprep_buffer(er_buffer, imb.shape[0]))
        # compute model cost and apply update
        if (n_updates % 2) == 0:
            if use_carry:
                # add examples from the carry buffer to this batch
                imb_fuzz = np.concatenate([imb_fuzz, cb_fuzz], axis=0)
            g_result = g_train_func(imb_fuzz, z0)
            g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:6], g_epoch_costs)]
            batch_obs_costs = g_result[8]
            batch_layer_klds = g_result[9]
            epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
            vae_nlls.append(1.*g_result[4])
            vae_klds.append(1.*g_result[5])
            gen_grad_norms.append(1.*g_result[6])
            inf_grad_norms.append(1.*g_result[7])
            # load the most difficult VAE inputs into the carry buffer
            vae_cost_rank = np.argsort(-1.0 * batch_obs_costs)
            full_batch = np.concatenate([imb, carry_buffer], axis=0)
            for i in range(carry_count):
                carry_buffer[i,:] = full_batch[vae_cost_rank[i],:]
                carry_costs.append(batch_obs_costs[vae_cost_rank[i]])
            g_batch_count += 1
            # process a validation batch
            if v_batch_count < 50:
                v_result = g_train_func(vmb_fuzz, z0)
                v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:6], v_epoch_costs)]
                v_batch_count += 1
        else:
            if use_er:
                d_result = d_train_func(imb_fuzz, z0, xer)
            else:
                d_result = d_train_func(imb_fuzz, z0)
            d_epoch_costs = [(v1 + v2) for v1, v2 in zip(d_result, d_epoch_costs)]
            d_batch_count += 1
        n_updates += 1
        # update experience replay buffer (a better update schedule may be helpful)
        if ((n_updates % (min(10,epoch)*20)) == 0) and use_er:
            update_exprep_buffer(er_buffer, gen_network, replace_frac=0.10)
    if (epoch == 30) or (epoch == 60):
        # cut learning rate in half
        lr = lrt.get_value(borrow=False)
        lr = lr / 2.0
        lrt.set_value(floatX(lr))
    if epoch > niter:
        # linearly decay learning rate
        lr = lrt.get_value(borrow=False)
        remaining_epochs = (niter + niter_decay + 1) - epoch
        lrt.set_value(floatX(lr - (lr / remaining_epochs)))
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    gen_grad_norms = np.asarray(gen_grad_norms)
    inf_grad_norms = np.asarray(inf_grad_norms)
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    d_epoch_costs = [(c / d_batch_count) for c in d_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}:".format(epoch)
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(g_bc_idx[:6], g_bc_names[:6])]
    str2 = " ".join(g_bc_strs)
    d_bc_strs = ["{0:s}: {1:.2f},".format(c_name, d_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(d_bc_idx[:4], d_bc_names[:4])]
    str3 = " ".join(d_bc_strs)
    ggn_qtiles = np.percentile(gen_grad_norms, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](ggn): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            ggn_qtiles[0], ggn_qtiles[1], ggn_qtiles[2], ggn_qtiles[3], np.max(gen_grad_norms))
    ign_qtiles = np.percentile(inf_grad_norms, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](ign): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            ign_qtiles[0], ign_qtiles[1], ign_qtiles[2], ign_qtiles[3], np.max(inf_grad_norms))
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str6 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str7 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    str8 = "    [min, mean, max](carry_costs): {0:.2f}, {1:.2f}, {2:.2f}".format( \
            min(carry_costs), sum(carry_costs)/len(carry_costs), max(carry_costs))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str9 = "    module kld -- {}".format(" ".join(kld_strs))
    str10 = "    validation -- nll: {0:.2f}, kld: {1:.2f}".format( \
            v_epoch_costs[4], v_epoch_costs[5])
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8, str9, str10])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    # generate some samples from the model prior
    samples = np.asarray(sample_func(sample_z0mb))
    color_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(sample_dir, epoch))
    # test reconstruction performance (inference + generation)
    if epoch < gauss_blur_weights.shape[0]:
        w_x = gauss_blur_weights[epoch]
    else:
        w_x = 1.0
    w_g = 1.0 - w_x
    tr_rec_batch = np.concatenate([carry_buffer, Xtr[0:100,:]], axis=0)
    tr_rec_batch = tr_rec_batch[0:100,:]
    va_rec_batch = Xva[0:100,:]
    if use_annealing and (w_x < 0.999):
        # add noise to reconstruction targets (if we trained with noise)
        tr_rb_fuzz = np.clip(gauss_blur(tr_rec_batch, Xtr_std, w_x, w_g),
                             a_min=-1.0, a_max=1.0)
        va_rb_fuzz = np.clip(gauss_blur(va_rec_batch, Xtr_std, w_x, w_g),
                             a_min=-1.0, a_max=1.0)
    else:
        # otherwise, use noise-free reconstruction targets
        tr_rb_fuzz = tr_rec_batch
        va_rb_fuzz = va_rec_batch
    # get the model reconstructions
    tr_rb_fuzz = train_transform(tr_rb_fuzz)
    va_rb_fuzz = train_transform(va_rb_fuzz)
    tr_recons = recon_func(tr_rb_fuzz)
    va_recons = recon_func(va_rb_fuzz)
    # stripe data for nice display (each reconstruction next to its target)
    tr_vis_batch = np.zeros((200, nc, npx, npx))
    va_vis_batch = np.zeros((200, nc, npx, npx))
    for rec_pair in range(100):
        idx_in = 2*rec_pair
        idx_out = 2*rec_pair + 1
        tr_vis_batch[idx_in,:,:,:] = tr_rb_fuzz[rec_pair,:,:,:]
        tr_vis_batch[idx_out,:,:,:] = tr_recons[rec_pair,:,:,:]
        va_vis_batch[idx_in,:,:,:] = va_rb_fuzz[rec_pair,:,:,:]
        va_vis_batch[idx_out,:,:,:] = va_recons[rec_pair,:,:,:]
    # draw images...
    color_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(sample_dir, epoch))
    color_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(sample_dir, epoch))







##############
# EYE BUFFER #
##############
