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
desc = 'test_van_vae_gan_annealed'
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
b1 = 0.5          # momentum term of adam
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
use_annealing = True # whether to anneal the target distribution while training

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
    out_shape=(ngf*4, 2, 2),
    fc_dim=ngfc,
    num_layers=2,
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
    use_fc=True,
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
    use_conv=True,
    mod_name='im_mod_3'
) # merge input to td_mod_3 and output of bu_mod_3, to place a distribution
  # over the rand_vals used in td_mod_3.

im_module_5 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*1),
    use_conv=True,
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
    num_layers=1,
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
lam_vae = sharedX(np.ones((1,)).astype(theano.config.floatX))
lam_kld = sharedX(np.ones((1,)).astype(theano.config.floatX))
obs_logvar = sharedX(np.zeros((1,)).astype(theano.config.floatX))
bounded_logvar = 6.0 * tanh((1.0/6.0) * obs_logvar)
g_params = [obs_logvar] + inf_gen_model.params

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
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# run an inference and reconstruction pass through the generative stuff
Xg_recon, kld_dicts = inf_gen_model.apply_im(Xg)
# feed reconstructions and their instigators into the discriminator.
# -- these values are used for training the generative stuff ONLY
Hg_recon, Yg_recon = disc_network.apply(input=Xg_recon, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)
Hg_world, Yg_world = disc_network.apply(input=Xg, ret_vals=ret_vals,
                                        ret_acts=True, app_sigm=True)

vae_layer_nlls = []
for hg_world, hg_recon in zip(Hg_world, Hg_recon):
    lnll = -1. * log_prob_gaussian(T.flatten(hg_world,2), T.flatten(hg_recon,2),
                                   log_vars=bounded_logvar[0], do_sum=False)
    vae_layer_nlls.append(T.mean(T.sum(lnll, axis=1)))
vae_nll_cost = vae_layer_nlls[0] #sum(vae_layer_nlls)

# KL-divergence part of cost
kld_tuples = [(mod_name, mod_kld) for mod_name, mod_kld in kld_dicts.items()]
obs_klds = [T.sum(tup[1], axis=1) for tup in kld_tuples]  # per-obs KLd for each latent layer
vae_layer_klds = [T.mean(kld_i) for kld_i in obs_klds]    # mean KLd for each latent layer
vae_kld_cost = sum(vae_layer_klds)                        # mean total KLd
# parameter regularization part of cost
vae_reg_cost = 1e-6 * sum([T.sum(p**2.0) for p in g_params])
# combined cost for generator stuff
vae_cost = vae_nll_cost + (lam_kld[0] * vae_kld_cost) + vae_reg_cost


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
# compute classification parts of GAN cost (for generator and discriminator)
gan_layer_nlls_world = []
gan_layer_nlls_model = []
gan_layer_nlls_gnrtr = []
weights = [1. for yd_world in Yd_world]
for yd_world, yd_model, w in zip(Yd_world, Yd_model, weights):
    lnll_world = bce(yd_world, T.ones_like(yd_world))
    lnll_model = bce(yd_model, T.zeros_like(yd_model))
    lnll_gnrtr = bce(yd_model, T.ones_like(yd_model))
    gan_layer_nlls_world.append(w * T.mean(lnll_world))
    gan_layer_nlls_model.append(w * T.mean(lnll_model))
    gan_layer_nlls_gnrtr.append(w * T.mean(lnll_gnrtr))
gan_nll_cost_world = sum(gan_layer_nlls_world)
gan_nll_cost_model = sum(gan_layer_nlls_model)
gan_nll_cost_gnrtr = sum(gan_layer_nlls_gnrtr)

# parameter regularization parts of GAN cost
gan_reg_cost_d = 1e-6 * sum([T.sum(p**2.0) for p in d_params])
gan_reg_cost_g = 1e-6 * sum([T.sum(p**2.0) for p in g_params])
# compute GAN cost for discriminator
gan_cost_d = gan_nll_cost_world + gan_nll_cost_model + gan_reg_cost_d
# compute GAN cost for generator
gan_cost_g = gan_nll_cost_gnrtr + gan_reg_cost_g

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################
full_cost_d = gan_cost_d
full_cost_g = gan_cost_g + (lam_vae[0] * vae_cost)

# stuff for performing updates
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)
g_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)

# build training cost and update functions
t = time()
print("Computing gradients...")
d_updates = d_updater(d_params, full_cost_d)
g_updates = g_updater(g_params, full_cost_g)
updates = d_updates + g_updates
print("Compiling sampling and reconstruction functions...")
Xtr_rec = train_transform(Xtr[0:200,:])
color_grid_vis(draw_transform(Xtr_rec), (10, 20), "{}/Xtr_rec.png".format(sample_dir))
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(Xtr_rec) # cheeky model implementation test
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost_g, gan_cost_g, vae_cost, vae_nll_cost, vae_kld_cost]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost_g', 'gan_cost_g', 'vae_cost', 'vae_nll_cost', 'vae_kld_cost']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg, Z0], g_cost_outputs, updates=g_updates)
# collect costs for discriminator parameters
d_basic_costs = [full_cost_d, gan_cost_d, gan_nll_cost_world, gan_nll_cost_model]
d_bc_idx = range(0, len(d_basic_costs))
d_bc_names = ['full_cost_d', 'gan_cost_d', 'gan_nll_cost_world', 'gan_nll_cost_model']
d_cost_outputs = d_basic_costs
# compile function for computing discriminator costs and updates
d_train_func = theano.function([Xd, Z0], d_cost_outputs, updates=d_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(sample_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_epochs = 0
n_updates = 0
t = time()
gauss_blur_weights = np.linspace(0.0, 1.0, 20) # weights for distribution "annealing"
sample_z0mb = rand_gen(size=(200, nz0))        # root noise for visualizing samples
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    vae_scale = 0.0003
    kld_scale = min(0.2, 0.02*epoch)
    lam_vae.set_value(np.asarray([vae_scale]).astype(theano.config.floatX))
    lam_kld.set_value(np.asarray([kld_scale]).astype(theano.config.floatX))
    g_epoch_costs = [0. for i in range(len(g_cost_outputs))]
    d_epoch_costs = [0. for i in range(len(d_cost_outputs))]
    g_batch_count = 0.
    d_batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        if epoch < gauss_blur_weights.shape[0]:
            w_x = gauss_blur_weights[epoch]
        else:
            w_x = 1.0
        w_g = 1.0 - w_x
        if use_annealing and (w_x < 0.999):
            imb = gauss_blur(imb, Xtr_std, w_x, w_g)
        imb = train_transform(imb)
        z0 = rand_gen(size=(nbatch, nz0))
        # compute model cost and apply update
        if (n_updates % 2) == 0:
            g_result = g_train_func(imb, z0)
            g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
            g_batch_count += 1
        else:
            d_result = d_train_func(imb, z0)
            d_epoch_costs = [(v1 + v2) for v1, v2 in zip(d_result, d_epoch_costs)]
            d_batch_count += 1
        n_updates += 1
        # if batch_count == 1000:
        #     print(" ")
        #     break
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    d_epoch_costs = [(c / d_batch_count) for c in d_epoch_costs]
    str1 = "Epoch {}:".format(epoch)
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
    str2 = " ".join(g_bc_strs)
    d_bc_strs = ["{0:s}: {1:.2f},".format(c_name, d_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(d_bc_idx, d_bc_names)]
    str3 = " ".join(d_bc_strs)
    joint_str = "\n".join([str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    n_epochs += 1
    # generate some samples from the model prior
    samples = np.asarray(sample_func(sample_z0mb))
    color_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(sample_dir, n_epochs))
    # sample some reconstructions from the model
    test_recons = recon_func(Xtr_rec)
    color_grid_vis(draw_transform(test_recons), (10, 20), "{}/rec_{}.png".format(sample_dir, n_epochs))
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))







##############
# EYE BUFFER #
##############
