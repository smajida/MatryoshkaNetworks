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
from lib.costs import log_prob_bernoulli, log_prob_gaussian
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import \
    shuffle, iter_data, get_masked_data, get_downsampling_masks

#
# Phil's business
#
from ModelBuilders import build_faces_cond_res
from MatryoshkaModules import BasicConvModule
from MatryoshkaNetworks import SimpleMLP

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./lsun"

scene_type = 'church'

# setup paths for dumping diagnostic info
desc = 'test_{}_adversarial_maxnorm50_3xKL_2step'.format(scene_type)
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file_1 = "{}/inf_gen_params_1.pkl".format(result_dir)
inf_gen_param_file_2 = "{}/inf_gen_params_2.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# locations of 64x64 faces dataset -- stored as a collection of .npy files
# data_dir = "{}/data".format(EXP_DIR)
if scene_type == 'church':
    data_dir = '/NOBACKUP/lsun/church_outdoor_train_center_crop'
elif scene_type == 'tower':
    data_dir = '/NOBACKUP/lsun/tower_train_center_crop'
elif scene_type == 'bridge':
    data_dir = '/NOBACKUP/lsun/bridge_train_center_crop'
elif scene_type == 'bedroom':
    data_dir = '/NOBACKUP/lsun/bedroom_train_center_crop'
else:
    raise Exception('Unknown scene type: {}'.format(scene_type))

# get a list of the .npy files that contain images in this directory. there
# shouldn't be any other files in the directory (hackish, but easy).
data_files = os.listdir(data_dir)
data_files.sort()
data_files = ["{}/{}".format(data_dir, file_name) for file_name in data_files]


def scale_to_tanh_range(X):
    """
    Scale the given 2d array to be in tanh range (i.e. -1...1).
    """
    X = X - np.min(X)
    X = X / np.max(X)
    X = 2. * (X - 0.5)
    X_std = np.std(X, axis=0, keepdims=True)
    return X, X_std


def load_and_scale_data(npy_file_name):
    """
    Load and scale data from the given npy file, and compute standard deviation
    too, to use when doing distribution annealing.
    """
    np_ary = np.load(npy_file_name)
    np_ary = np_ary.astype(theano.config.floatX)
    X, X_std = scale_to_tanh_range(np_ary)
    return X, X_std


def train_transform(X, add_fuzz=True):
    # transform vectorized observations into convnet inputs
    if add_fuzz:
        X = X + ((2. / 256.) * npr.uniform(size=X.shape))
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3))


def draw_transform(X):
    # transform vectorized observations into drawable images
    X = X - np.min(X)
    X = X / np.max(X)
    X = 255. * X
    return floatX(X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1))


def rand_gen(size, noise_type='normal'):
    if noise_type == 'normal':
        r_vals = floatX(np_rng.normal(size=size))
    elif noise_type == 'uniform':
        r_vals = floatX(np_rng.uniform(size=size, low=-1.0, high=1.0))
    else:
        assert False, "unrecognized noise type!"
    return r_vals


def rand_fill(x, m, scale=1.):
    '''
    Fill masked parts of x, indicated by m, using uniform noise.
    -- assume data is in [-1, 1] (i.e. comes from train_transform())
    '''
    m = 1. * (m > 1e-3)
    nz = (scale * (np_rng.uniform(size=x.shape) - 0.5))
    x_nz = (m * nz) + ((1. - m) * x)
    return x_nz


def load_data_file(df_name, va_count=2000):
    # load all data into memory
    print('loading data from {}...'.format(df_name))
    xtr, x_std = load_and_scale_data(df_name)
    # split data into validation and training bits based
    # on placement within the data file.
    xva = xtr[:va_count, :]
    xtr = xtr[va_count:, :]
    # shuffle training/validation data and compute mean
    xtr = shuffle(xtr)
    xva = shuffle(xva)
    xmu = np.mean(xtr, axis=0)
    print('done.')
    return xtr, xva, xmu

# load an initial portion of data
Xtr, Xva, Xmu = load_data_file(data_files[0])


set_seed(123)      # seed for shared rngs
nc = 3             # # of channels in image
nbatch = 50        # # of examples in batch
npx = 64           # # of pixels width/height of images
nz0 = 64           # # of dim for Z0
nz1 = 4            # # of dim for Z1
ngf = 32           # base # of filters for conv layers in generative stuff
ngfc = 256         # # of filters in fully connected layers of generative stuff
nx = npx * npx * nc  # # of dimensions in X
niter = 150        # # of iter at starting learning rate
niter_decay = 250  # # of iter to linearly decay learning rate to zero
multi_rand = True  # whether to use stochastic variables at multiple scales
use_conv = True    # whether to use "internal" conv layers in gen/disc networks
use_bn = False     # whether to use batch normalization throughout the model
act_func = 'lrelu'  # activation func to use where they can be selected
use_td_cond = False
kld_weight = 3.
depth_8x8 = 1
depth_16x16 = 1
depth_32x32 = 1


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


# construct the "wrapper" model for managing all our modules
inf_gen_model_1 = \
    build_faces_cond_res(
        nc=nc, nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc,
        use_bn=use_bn, act_func='lrelu', use_td_cond=use_td_cond,
        depth_8x8=depth_8x8, depth_16x16=depth_16x16, depth_32x32=depth_32x32)
inf_gen_model_2 = \
    build_faces_cond_res(
        nc=nc, nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc,
        use_bn=use_bn, act_func='lrelu', use_td_cond=use_td_cond,
        depth_8x8=depth_8x8, depth_16x16=depth_16x16, depth_32x32=depth_32x32)

# setup a simple down-sampling convolutional net to act as the
# "distributional adversary" -- modules listed from bottom to top

# (64, 64) -> (64, 64)
ac_mod_1 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=3,
        out_chans=32,
        stride='single',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_1')

# (64, 64) -> (64, 64)
ac_mod_2 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=32,
        out_chans=32,
        stride='single',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_2')

# (64, 64) -> (32, 32)
ac_mod_3 = \
    BasicConvModule(
        filt_shape=(2, 2),
        in_chans=32,
        out_chans=64,
        stride='double',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_3')

# (32, 32) -> (32, 32)
ac_mod_4 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=64,
        out_chans=64,
        stride='single',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_4')

# (32, 32) -> (16, 16)
ac_mod_5 = \
    BasicConvModule(
        filt_shape=(2, 2),
        in_chans=64,
        out_chans=96,
        stride='double',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_5')

# (16, 16) -> (16, 16)
ac_mod_6 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=96,
        out_chans=96,
        stride='double',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_6')

# (16, 16) -> (8, 8)
ac_mod_7 = \
    BasicConvModule(
        filt_shape=(2, 2),
        in_chans=96,
        out_chans=128,
        stride='double',
        apply_bn=False,
        act_func='lrelu',
        mod_name='ac_mod_7')

ac_modules = [ac_mod_1, ac_mod_2, ac_mod_3, ac_mod_4,
              ac_mod_5, ac_mod_6, ac_mod_7]
ac_cost_layers = ['ac_mod_5', 'ac_mod_7']
ac_cost_weights = [0.1, 0.1]

adv_conv = SimpleMLP(modules=ac_modules)


# grab data to feed into the model
def make_model_input(x_in, x_mu):
    # construct "imputational upsampling" masks
    xg_gen, xg_inf, xm_gen = \
        get_masked_data(x_in, im_shape=(nc, npx, npx), drop_prob=0.,
                        occ_shape=(20, 20), occ_count=3,
                        data_mean=x_mu)
    # reshape and process data for use as model input
    xm_gen = 1. - xm_gen  # mask is 1 for unobserved pixels
    xm_inf = xm_gen       # mask is 1 for pixels to predict
    xg_gen = train_transform(xg_gen)
    xm_gen = train_transform(xm_gen, add_fuzz=False)
    xg_inf = train_transform(xg_inf)
    xm_inf = train_transform(xm_inf, add_fuzz=False)
    return xg_gen, xm_gen, xg_inf, xm_inf


def obs_fix(obs_conv, max_norm=10.):
    obs_flat = T.flatten(obs_conv, 2)
    obs_cent = obs_flat - T.mean(obs_flat, axis=1, keepdims=True)
    norms = T.sqrt(T.sum(obs_cent**2., axis=1, keepdims=True))
    rescale = T.minimum((max_norm / norms), 1.)
    obs_bnd_norm = rescale * obs_cent
    return obs_bnd_norm

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
log_var = sharedX(floatX([0.0]))
X_init = sharedX(floatX(np.zeros((1, nc, npx, npx))))
gen_params = inf_gen_model_1.gen_params + inf_gen_model_2.gen_params
inf_params = inf_gen_model_1.inf_params + inf_gen_model_2.inf_params
all_params = inf_gen_model_1.all_params + inf_gen_model_2.all_params + [log_var, X_init]
adv_params = adv_conv.params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg_gen = T.tensor4()  # input to generator, with some parts masked out
Xm_gen = T.tensor4()  # mask indicating parts that are masked out
Xg_inf = T.tensor4()  # complete observation, for input to inference net
Xm_inf = T.tensor4()  # mask for which bits to predict

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################

# parameter regularization part of cost
vae_reg_cost = 1e-5 * (sum([T.sum(p**2.0) for p in all_params]) +
                       sum([T.sum(p**2.0) for p in adv_params]))

# feed all masked inputs through the inference network
Xa_gen_1 = T.concatenate([Xg_gen, Xm_gen], axis=1)
Xa_inf_1 = T.concatenate([Xg_gen, Xm_gen, Xg_inf, Xm_inf], axis=1)
im_res_dict_1 = inf_gen_model_1.apply_im(input_gen=Xa_gen_1, input_inf=Xa_inf_1)
kld_dict_1 = im_res_dict_1['kld_dict']
Xg_recon_1 = im_res_dict_1['output']
Xg_recon_1 = tanh(Xg_recon_1)

# apply occlusion mask to get the full imputed image from first step
Xm_inf_mask = 1. * (Xm_inf > 1e-3)
Xg_guess_1 = (Xm_inf_mask * Xg_recon_1) + ((1. - Xm_inf_mask) * Xg_gen)
#               -- imputed values --          -- known values --

# run the second step of reconstruction (input is first stage reconstruction)
Xa_gen_2 = T.concatenate([Xg_guess_1, Xm_gen], axis=1)
Xa_inf_2 = T.concatenate([Xg_guess_1, Xm_gen, Xg_inf, Xm_inf], axis=1)
im_res_dict_2 = inf_gen_model_2.apply_im(input_gen=Xa_gen_2, input_inf=Xa_inf_2)
kld_dict_2 = im_res_dict_2['kld_dict']
Xg_recon_2 = im_res_dict_2['output'] + im_res_dict_1['output']
Xg_recon_2 = tanh(Xg_recon_2)

# apply occlusion mask to get the full imputed image from second step
Xg_guess_2 = (Xm_inf_mask * Xg_recon_2) + ((1. - Xm_inf_mask) * Xg_gen)
#               -- imputed values --          -- known values --


# compute pixel-level reconstruction error on missing pixels
# -- We'll bound the norms of the reconstructions and targets in both pixel
#    and adversarial spaces, to keep their errors sort of comparable.
x_truth = obs_fix(Xg_inf, max_norm=50.)
x_guess = obs_fix(Xg_guess_2, max_norm=50.)
pix_loss = T.sum(log_prob_gaussian(x_truth, x_guess,
                 log_vars=log_var[0], mask=T.flatten(Xm_inf_mask, 2),
                 use_huber=0.5, do_sum=False), axis=1)

# feed original observation and reconstruction into conv net
adv_dict_truth = adv_conv.apply(Xg_inf, return_dict=True)
adv_dict_guess = adv_conv.apply(Xg_guess_2, return_dict=True)

# compute adversarial reconstruction losses
adv_losses = []
adv_act_regs = []
for (ac_cost_layer, ac_cost_weight) in zip(ac_cost_layers, ac_cost_weights):
    # apply tanh for a quick-and-dirty bound on loss
    x_truth = obs_fix(adv_dict_truth[ac_cost_layer], max_norm=50.)
    x_guess = obs_fix(adv_dict_guess[ac_cost_layer], max_norm=50.)
    # compute adversarial distribution matching cost
    acl_log_p_x = T.sum(log_prob_gaussian(x_truth, x_guess,
                        log_vars=log_var[0], use_huber=0.5,
                        do_sum=False), axis=1)
    adv_losses.append(ac_cost_weight * acl_log_p_x)
    acl_act_reg = T.sum(T.sqr(adv_dict_truth[ac_cost_layer])) + \
        T.sum(T.sqr(adv_dict_guess[ac_cost_layer]))
    adv_act_regs.append(acl_act_reg)
log_p_x = (0.9 * sum(adv_losses)) + (0.1 * pix_loss)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# convert KL dict to aggregate KLds over inference steps
kl_by_td_mod = {tdm_name: (kld_dict_1[tdm_name] + kld_dict_2[tdm_name]) for
                tdm_name in kld_dict_1.keys()}
# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, mod_kld) for mod_name, mod_kld in kl_by_td_mod.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]

# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

# compute the KLd cost to use for optimization
opt_kld_cost = lam_kld[0] * vae_kld_cost

# combined cost for generator stuff
vae_cost = vae_nll_cost + vae_kld_cost
vae_obs_costs = vae_obs_nlls + vae_obs_klds
# cost used by the optimizer
full_cost = vae_nll_cost + opt_kld_cost + vae_reg_cost
adv_cost = -full_cost + (0.03 * sum(adv_act_regs))

#
# test the model implementation
#
inputs = [Xg_gen, Xm_gen, Xg_inf, Xm_inf]
outputs = [full_cost]
print('Compiling test function...')
test_func = theano.function(inputs, outputs)
# test the model implementation
model_input = make_model_input(Xtr[0:100, :], Xmu)
test_out = test_func(*model_input)
print('DONE.')

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

# stuff for performing updates
lrt = sharedX(0.0005)
b1t = sharedX(0.9)
updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=100.0)
# for adversary
adv_lrt = sharedX(0.0)
adv_updater = updates.Adam(lr=adv_lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=100.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
all_updates = updater(all_params, full_cost, return_grads=False)
adv_updates = adv_updater(adv_params, adv_cost, return_grads=False)
jnt_updates = all_updates + adv_updates

print("Compiling sampling and reconstruction functions...")
# sampling requires a wrapper around the reconstruction function which
# flips the CondInfGen model's "sample source" switch.
recon_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], Xg_guess_2)


def sample_func(xg_gen, xm_gen, model1, model2):
    '''
    switchy samply wrapper funk.
    '''
    model1.set_sample_switch(source='gen')
    model2.set_sample_switch(source='gen')
    x_out = recon_func(xg_gen, xm_gen, xg_gen, xm_gen)
    model1.set_sample_switch(source='inf')
    model2.set_sample_switch(source='inf')
    # get the blended input and predictions for missing pixels
    xm_gen = 1. * (xm_gen > 1e-3)
    x_out = (xm_gen * x_out) + ((1. - xm_gen) * xg_gen)
    return x_out

# test samplers for conditional generation
xg_gen, xm_gen, xg_inf, xm_inf = make_model_input(Xtr[:100, :], Xmu)
xg_rec = recon_func(xg_gen, xm_gen, xg_inf, xm_inf)
xg_rec = sample_func(xg_gen, xm_gen, inf_gen_model_1, inf_gen_model_2)

print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost, full_cost, vae_cost, vae_nll_cost,
                 vae_kld_cost, vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost', 'full_cost', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], g_cost_outputs,
                               updates=jnt_updates)
g_eval_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], g_cost_outputs)
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

batches_per_epoch = 1000
t = time()
for epoch in range(1, (niter + niter_decay + 1)):
    # load a file containing a subset of the large full training set
    df_num = (epoch - 1) % len(data_files)
    Xtr, Xva, Xmu = load_data_file(data_files[df_num])
    epoch_batch_count = Xtr.shape[0] // nbatch
    # mess with the KLd cost
    lam_kld.set_value(floatX([kld_weight]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0
    v_batch_count = 0
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=epoch_batch_count):
        # set adversary to be slow relative to generator...
        adv_lr = 0.05 * lrt.get_value(borrow=False)
        adv_lrt.set_value(floatX(adv_lr))
        # transform training batch to model input format
        imb_input = make_model_input(imb, Xmu)
        # compute loss and apply updates for this batch
        g_result = g_train_func(*imb_input)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        batch_obs_costs = g_result[5]
        batch_layer_klds = g_result[6]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # run a smallish number of validation batches per epoch
        if v_batch_count < 25:
            vmb = Xva[v_batch_count * nbatch:(v_batch_count + 1) * nbatch, :]
            vmb_input = make_model_input(vmb, Xmu)
            v_result = g_eval_func(*vmb_input)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:5], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 10) or (epoch == 25) or (epoch == 50) or (epoch == 100):
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
    inf_gen_model_1.dump_params(inf_gen_param_file_1)
    inf_gen_model_2.dump_params(inf_gen_param_file_2)
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    v_bc_strs = ["{0:s}: {1:.2f},".format(c_name, v_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str3 = " ".join(v_bc_strs)
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str6 = "    module kld -- {}".format(" ".join(kld_strs))
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
    ######################
    # DRAW SOME PICTURES #
    ######################
    if (epoch < 20) or (((epoch - 1) % 10) == 0):
        # sample some reconstructions directly from the conditional model
        xg_gen, xm_gen, xg_inf, xm_inf = make_model_input(Xva[:100, :], Xmu)
        xg_rec = sample_func(xg_gen, xm_gen, inf_gen_model_1, inf_gen_model_2)
        # put noise in missing region of xg_gen
        xg_gen = rand_fill(xg_gen, xm_gen, scale=0.2)
        # stripe data for nice display (each reconstruction next to its target)
        vis_batch = np.zeros((200, nc, npx, npx))
        for rec_pair in range(100):
            idx_in = 2 * rec_pair
            idx_out = 2 * rec_pair + 1
            vis_batch[idx_in, :, :, :] = xg_gen[rec_pair, :, :, :]
            vis_batch[idx_out, :, :, :] = xg_rec[rec_pair, :, :, :]
        # draw images...
        color_grid_vis(draw_transform(vis_batch), (10, 20), "{}/gen_va_{}.png".format(result_dir, epoch))


##############
# EYE BUFFER #
##############
