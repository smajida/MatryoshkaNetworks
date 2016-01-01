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
from MatryoshkaNetworks import InfGenModel, DiscNetworkGAN

# path for dumping experiment info and fetching dataset
EXP_DIR = "./svhn"
DATA_SIZE = 250000

# setup paths for dumping diagnostic info
desc = 'test_resnet_van'
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
tr_file = "{}/data/svhn_train.pkl".format(EXP_DIR)
te_file = "{}/data/svhn_test.pkl".format(EXP_DIR)
ex_file = "{}/data/svhn_extra.pkl".format(EXP_DIR)
# load dataset (load more when using adequate computers...)
data_dict = load_svhn(tr_file, te_file, ex_file=ex_file, ex_count=DATA_SIZE)

# stack data into a single array and rescale it into [-1,1]
Xtr = np.concatenate([data_dict['Xtr'], data_dict['Xte'], data_dict['Xex']], axis=0)
del data_dict
Xtr = Xtr - np.min(Xtr)
Xtr = Xtr / np.max(Xtr)
Xtr = 2.0 * (Xtr - 0.5)
Xtr_std = np.std(Xtr, axis=0, keepdims=True)
Xtr_var = Xtr_std**2.0

set_seed(1)       # seed for shared rngs
l2 = 1.0e-5       # l2 weight decay
b1 = 0.9          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 100      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz0 = 64          # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngfc = 256        # # of filters in fully connected layers of generative stuff
ngf = 64          # base # of filters for conv layers in generative stuff
ndfc = 256        # # of filters in fully connected layers of discriminator
ndf = 64          # base # of filters for conv layers in discriminator
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
multi_rand = True # whether to use stochastic variables at multiple scales
multi_disc = True # whether to use discriminator feedback at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_vae_cost = True # whether to use VAE-style or GAN-style distribution matching

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
    out_shape=(ngf*4, 2, 2),
    fc_dim=ngfc,
    num_layers=1,
    apply_bn=True,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 2, 2)

td_module_2 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=False,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 4, 4)

td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=ngf,
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
    conv_chans=ngf,
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=False,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_4'
) # output is (batch, ngf*2, 16, 16)

td_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=ngf,
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

bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=True,
    stride='single',
    act_func='relu',
    mod_name='bu_mod_6'
) # output is (batch, ngf*1, 32, 32)

bu_module_5 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=ngf,
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_5'
) # output is (batch, ngf*2, 16, 16)

bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=ngf,
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_4'
) # output is (batch, ngf*2, 8, 8)

bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=ngf,
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 4, 4)

bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_2'
) # output is (batch, ngf*4, 2, 2)

bu_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*2*2),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=False,
    mod_name='bu_mod_1'
) # output is (batch, nz0), (batch, nz0)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_6, bu_module_5, bu_module_4,
              bu_module_3, bu_module_2, bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=False,
    mod_name='im_mod_3'
) # merge input to td_mod_3 and output of bu_mod_3, to place a distribution
  # over the rand_vals used in td_mod_3.

im_module_5 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*1),
    use_conv=False,
    mod_name='im_mod_5'
) # merge input to td_mod_5 and output of bu_mod_5, to place a distribution
  # over the rand_vals used in td_mod_5.

im_modules = [im_module_3, im_module_5]

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
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'},
}

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=tanh
)

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
    stride='double',
    act_func='lrelu',
    mod_name='disc_mod_1'
) # output is (batch, ndf*1, 16, 16)

disc_module_2 = \
DiscConvResModule(
    in_chans=(ndf*1),
    out_chans=(ndf*2),
    conv_chans=ndf,
    filt_shape=(3,3),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_2'
) # output is (batch, ndf*2, 8, 8)

disc_module_3 = \
DiscConvResModule(
    in_chans=(ndf*2),
    out_chans=(ndf*4),
    conv_chans=ndf,
    filt_shape=(3,3),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_3'
) # output is (batch, ndf*2, 4, 4)

disc_module_4 = \
DiscConvResModule(
    in_chans=(ndf*4),
    out_chans=(ndf*4),
    conv_chans=(ndf*2),
    filt_shape=(3,3),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_4'
) # output is (batch, ndf*4, 2, 2)

disc_module_5 = \
DiscFCModule(
    fc_dim=ndfc,
    in_dim=(ndf*4*2*2),
    num_layers=2,
    apply_bn=True,
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
lam_kld = sharedX(np.ones((1,)).astype(theano.config.floatX))
obs_logvar = sharedX(np.zeros((1,)).astype(theano.config.floatX))
bounded_logvar = 6.0 * tanh((1.0/6.0) * obs_logvar)
g_params = [obs_logvar] + inf_gen_model.params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

if multi_disc:
    # multi-scale discriminator guidance
    ret_vals = range(1,len(disc_network.modules))
else:
    # full-scale discriminator guidance only
    ret_vals = [ (len(disc_network.modules)-1) ]

# Setup symbolic vars for the model inputs, outputs, and costs
Xd = T.tensor4()  # symbolic var for inputs to discriminator
Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##############################################
# CONSTRUCT COST VARIABLES FOR THE GENERATOR #
##############################################
# run an inference and reconstruction pass through the generative stuff
Xg_recon, kld_dicts = inf_gen_model.apply_im(Xg)
# feed reconstructions and their instigators into the discriminator.
# -- these values are used for training the generative stuff ONLY
Hg_recon, Yg_recon = disc_network.apply(input=Xg_recon, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
Hg_world, Yg_world = disc_network.apply(input=Xg, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
if use_vae_cost:
    print("Using VAE-style distribution matching.")
    # VAE-style distribution matching tries to reconstruct the particular
    # distribution over discriminator bits for a given world input
    pass
else:
    print("Using GAN-style distribution matching.")
    # VAE-style distribution matching tries to reconstruct a generic (all ones)
    # distribution over discriminator bits for a given world input
    Yg_world = [T.ones_like(yg_recon) for yg_recon in Yg_recon]

g_layer_nlls = [(-1.0 * log_prob_bernoulli(yg_world, yg_recon)) \
                for (yg_world, yg_recon) in zip(Yg_world, Yg_recon)]
g_nll_cost = sum([T.mean(lnll) for lnll in g_layer_nlls])

# KL-divergence part of cost
kld_tuples = [(mod_name, mod_kld) for mod_name, mod_kld in kld_dicts.items()]
obs_klds = [T.sum(tup[1], axis=1) for tup in kld_tuples]  # per-obs KLd for each latent layer
layer_klds = [T.mean(kld_i) for kld_i in obs_klds]        # mean KLd for each latent layer
g_kld_cost = sum(layer_klds)                              # mean total KLd
# parameter regularization part of cost
g_reg_cost = 1e-6 * sum([T.sum(p**2.0) for p in g_params])
# combined cost for generator stuff
g_cost = g_nll_cost + (lam_kld[0] * g_kld_cost) + g_reg_cost


##################################################
# CONSTRUCT COST VARIABLES FOR THE DISCRIMINATOR #
##################################################
# run an un-grounded pass through generative stuff (i.e. sample from model)
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)
# feed "world-generated" data and "model-generated" data into the discriminator
Hd_model, Yd_model = disc_network.apply(input=Xd_model, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
Hd_world, Yd_world = disc_network.apply(input=Xd, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
# compute classification parts of discriminator cost
d_layer_nlls_world = [(-1.0 * log_prob_bernoulli(T.ones_like(yd_world), yd_world)) \
                      for yd_world in Yd_world]
d_nll_cost_world = sum([T.mean(lnll) for lnll in d_layer_nlls_world])
d_layer_nlls_model = [(-1.0 * log_prob_bernoulli(T.zeros_like(yd_model), yd_model)) \
                      for yd_model in Yd_model]
d_nll_cost_model = sum([T.mean(lnll) for lnll in d_layer_nlls_model])
# parameter regularization part of cost
d_reg_cost = 1e-6 * sum([T.sum(p**2.0) for p in d_params])
# compute full discriminator cost
d_cost = d_nll_cost_model + d_nll_cost_world + d_reg_cost

# compile a theano function strictly for sampling reconstructions
recon_func = theano.function([Xg], Xg_recon)
# TEMP TEST FOR MODEL ARCHITECTURE
Xtr_rec = train_transform(Xtr[0:200,:])
test_recons = recon_func(Xtr_rec)
color_grid_vis(draw_transform(Xtr_rec), (10, 20), "{}/Xtr_rec.png".format(sample_dir))

# stuff for performing updates
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)
g_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)

# build training cost and update functions
t = time()
print("Computing gradients...")
d_updates = d_updater(d_params, d_cost)
g_updates = g_updater(g_params, g_cost)
updates = d_updates + g_updates
print("Compiling sampling function...")
sample_func = theano.function([Z0], Xd_model)
print("Compiling training functions...")
g_cost_outputs = [g_cost, g_nll_cost, g_kld_cost, g_reg_cost] + layer_klds
g_train_func = theano.function([Xg], g_cost_outputs, updates=g_updates)
d_cost_outputs = [d_cost, d_nll_cost_world, d_nll_cost_model, d_reg_cost]
d_train_func = theano.function([Xd, Z0], d_cost_outputs, updates=d_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(sample_dir)
out_file = open(log_name, 'wb')

n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
sample_z0mb = rand_gen(size=(200, nz0)) # noise samples for top generator module
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    scale = min(0.1, 0.02*epoch)
    lam_kld.set_value(np.asarray([scale]).astype(theano.config.floatX))
    g_epoch_costs = [0. for i in range(len(g_cost_outputs))]
    d_epoch_costs = [0. for i in range(len(d_cost_outputs))]
    batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        imb = train_transform(imb)
        z0 = rand_gen(size=(nbatch, nz0))
        # compute model cost and apply update
        g_result = g_train_func(imb)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
        d_result = d_train_func(imb, z0)
        d_epoch_costs = [(v1 + v2) for v1, v2 in zip(d_result, d_epoch_costs)]
        batch_count += 1
        n_updates += 1
        n_examples += len(imb)
    g_epoch_costs = [(c / batch_count) for c in g_epoch_costs]
    str1 = "Epoch {}:".format(epoch)
    str2 = "    g_cost: {0:.4f}, nll_cost: {1:.4f}, kld_cost: {2:.4f}, reg_cost: {3:.4f}".format( \
            g_epoch_costs[0], g_epoch_costs[1], g_epoch_costs[2], g_epoch_costs[3])
    kld_strs = ["       "]
    for i, kld_i in enumerate(g_epoch_costs[4:]):
        kld_strs.append("{0:s}: {1:.4f},".format(kld_tuples[i][0], kld_i))
    str3 = " ".join(kld_strs)
    str4 = "    d_cost: {0:.4f}, nll_cost_world: {1:.4f}, nll_cost_model: {2:.4f}".format( \
            d_epoch_costs[0], d_epoch_costs[1], d_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str3, str4])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    n_epochs += 1
    # generate some samples from the model prior
    samples = np.asarray(sample_func(sample_z0mb))
    color_grid_vis(draw_transform(samples), (10, 20), "{}/{}_gen.png".format(sample_dir, n_epochs))
    # sample some reconstructions from the model
    test_recons = recon_func(Xtr_rec)
    color_grid_vis(draw_transform(test_recons), (10, 20), "{}/{}_rec.png".format(sample_dir, n_epochs))
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))







##############
# EYE BUFFER #
##############
