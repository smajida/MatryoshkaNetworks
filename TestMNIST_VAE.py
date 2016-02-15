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
                              GenTopModule, InfConvMergeModule, \
                              InfTopModule, BasicConvResModule, \
                              DiscConvResModule, DiscFCModule
from MatryoshkaNetworks import InfGenModel, DiscNetworkGAN, GenNetworkGAN

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_vae_relu_short_model_basic_kld_all_noise_015'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load binarized MNIST dataset
data_path = "{}/data/".format(EXP_DIR)
Xtr, Xva, Xte = load_binarized_mnist(data_path=data_path)
Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
Xva = Xte


set_seed(1)       # seed for shared rngs
nc = 1            # # of channels in image
nbatch = 200      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 64          # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngf = 64          # base # of filters for conv layers in generative stuff
ngfc = 256        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
niter = 400       # # of iter at starting learning rate
niter_decay = 400 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_bn = True     # whether to use batch normalization throughout the model
use_td_cond = False # whether to use top-down conditioning in generator
act_func = 'relu' # activation func to use where they can be selected
iwae_samples = 1 # number of samples to use in MEN bound
noise_std = 0.15  # amount of noise to inject in BU and IM modules
use_bu_noise = True
use_td_noise = True
mod_type = 0

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
GenTopModule(
    rand_dim=nz0,
    out_shape=(ngf*4, 7, 7),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='single',
    act_func=act_func,
    apply_bn=use_bn,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
td_module_2a = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='single',
    act_func=act_func,
    apply_bn=use_bn,
    mod_name='td_mod_2a'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (14, 14)
td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*4),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (14, 14)
td_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='single',
    act_func=act_func,
    apply_bn=use_bn,
    mod_name='td_mod_4'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (14, 14)
td_module_4a = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    stride='single',
    act_func=act_func,
    apply_bn=use_bn,
    mod_name='td_mod_4a'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (28, 28)
td_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    filt_shape=(3,3),
    use_rand=multi_rand,
    use_conv=use_conv,
    apply_bn=use_bn,
    act_func=act_func,
    us_stride=2,
    mod_name='td_mod_5'
) # output is (batch, ngf*1, 28, 28)

# (28, 28) -> (28, 28)
td_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='td_mod_6'
) # output is (batch, ngf*1, 28, 28)

# (28, 28) -> (28, 28)
td_module_7 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
    apply_bn=False,
    use_noise=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_7'
) # output is (batch, c, 28, 28)

# modules must be listed in "evaluation order"
#td_modules = [td_module_1, td_module_2, td_module_2a, td_module_3, td_module_4,
#              td_module_4a, td_module_5, td_module_6, td_module_7]
td_modules = [td_module_1, td_module_2, td_module_3, td_module_4,
              td_module_5, td_module_6, td_module_7]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

# (7, 7) -> FC
bu_module_1 = \
InfTopModule(
    bu_chans=(ngf*4*7*7),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
) # output is (batch, 2*nz0)

# (7, 7) -> (7, 7)
bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2'
) # output is (batch, ngf*4, 7, 7)

# (7, 7) -> (7, 7)
bu_module_2a = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_2a'
) # output is (batch, ngf*4, 7, 7)

# (14, 14) -> (7, 7)
bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 7, 7)

# (14, 14) -> (14, 14)
bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_4'
) # output is (batch, ngf*2, 14, 14)

# (14, 14) -> (14, 14)
bu_module_4a = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_4a'
) # output is (batch, ngf*2, 14, 14)

# (28, 28) -> (14, 14)
bu_module_5 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    use_conv=use_conv,
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_5'
) # output is (batch, ngf*2, 14, 14)

# (28, 28) -> (28, 28)
bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_6'
) # output is (batch, ngf*1, 28, 28)

# (28, 28) -> (28, 28)
bu_module_7 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=False,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_7'
) # output is (batch, ngf*1, 28, 28)

# modules must be listed in "evaluation order"
#bu_modules = [bu_module_7, bu_module_6, bu_module_5, bu_module_4a, bu_module_4,
#              bu_module_3, bu_module_2a, bu_module_2, bu_module_1]
bu_modules = [bu_module_7, bu_module_6, bu_module_5, bu_module_4,
              bu_module_3, bu_module_2, bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*4),
    use_conv=True,
    apply_bn=use_bn,
    use_td_cond=use_td_cond,
    mod_type=mod_type,
    act_func=act_func,
    mod_name='im_mod_3'
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
    mod_type=mod_type,
    act_func=act_func,
    mod_name='im_mod_5'
)


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
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'}
}

# construct the "wrapper" object for managing all our modules
output_transform = lambda x: sigmoid(T.clip(x, -15.0, 15.0))
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=output_transform
)

#inf_gen_model.load_params(inf_gen_param_file)

####################################
# Setup the optimization objective #
####################################
lam_vae = sharedX(floatX([1.0]))
lam_kld = sharedX(floatX([1.0]))
noise = sharedX(floatX([noise_std]))
gen_params = inf_gen_model.gen_params
inf_params = inf_gen_model.inf_params
g_params = gen_params + inf_params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in g_params])
if iwae_samples == 1:
    # run an inference and reconstruction pass through the generative stuff
    im_res_dict = inf_gen_model.apply_im(Xg, noise=noise)
    Xg_recon = im_res_dict['td_output']
    kld_dict = im_res_dict['kld_dict']
    log_p_z = sum(im_res_dict['log_p_z'])
    log_q_z = sum(im_res_dict['log_q_z'])

    log_p_x = T.sum(log_prob_bernoulli( \
                    T.flatten(Xg,2), T.flatten(Xg_recon,2),
                    do_sum=False), axis=1)

    # compute reconstruction error part of free-energy
    vae_obs_nlls = -1.0 * log_p_x
    vae_nll_cost = T.mean(vae_obs_nlls)

    # compute per-layer KL-divergence part of cost
    kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dict.items()]
    vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
    vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]
    # compute total per-observation KL-divergence part of cost
    vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
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
else:
    # run an inference and reconstruction pass through the generative stuff
    batch_size = Xg.shape[0]
    Xg_rep = T.extra_ops.repeat(Xg, iwae_samples, axis=0)
    im_res_dict = inf_gen_model.apply_im(Xg_rep, noise=noise)
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

    vae_obs_costs = -1.0 * T.sum((nis_weights * log_ws_mat), axis=1)

    # free-energy log likelihood bound...
    vae_cost = -1.0 * T.mean(log_mean_exp(log_ws_mat, axis=1))

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

    # costs used by the optimizer -- train on combined VAE/IWAE costs
    full_cost_gen = ((1.0 - lam_vae[0]) * T.mean(vae_obs_costs)) + \
                    (lam_vae[0] * (vae_nll_cost + vae_kld_cost)) + \
                    vae_reg_cost
    full_cost_inf = full_cost_gen

    # get simple reconstruction, for other purposes
    im_rd = inf_gen_model.apply_im(Xg, noise=noise)
    Xg_recon = im_rd['td_output']

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

# stuff for performing updates
lrt = sharedX(0.001)
b1t = sharedX(0.8)
gen_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)
inf_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)

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
test_recons = recon_func(train_transform(Xtr[0:100,:])) # cheeky model implementation test
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost_gen, full_cost_inf, vae_cost, vae_nll_cost,
                 vae_kld_cost, gen_grad_norm, inf_grad_norm,
                 vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost_gen', 'full_cost_inf', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'gen_grad_norm', 'inf_grad_norm',
              'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg], g_cost_outputs, updates=g_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
lam_vae.set_value(floatX([0.5]))
kld_weights = np.linspace(0.0,1.0,20)
sample_z0mb = rand_gen(size=(200, nz0)) # root noise for visualizing samples
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    Xva = shuffle(Xva)
    # mess with the KLd cost
    #if ((epoch-1) < len(kld_weights)):
    #    lam_kld.set_value(floatX([kld_weights[epoch-1]]))
    lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    gen_grad_norms = []
    inf_grad_norms = []
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    v_batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count)*nbatch
            vmb = Xva[start_idx:(start_idx+nbatch),:]
        else:
            vmb = Xva[0:nbatch,:]
        # transform noisy training batch and carry buffer to "image format"
        imb_img = train_transform(imb)
        vmb_img = train_transform(vmb)
        # train vae on training batch
        noise.set_value(floatX([noise_std]))
        g_result = g_train_func(imb_img.astype(theano.config.floatX))
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1.*g_result[3])
        vae_klds.append(1.*g_result[4])
        gen_grad_norms.append(1.*g_result[5])
        inf_grad_norms.append(1.*g_result[6])
        batch_obs_costs = g_result[7]
        batch_layer_klds = g_result[8]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            noise.set_value(floatX([0.0]))
            v_result = g_train_func(vmb_img)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:6], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 25) or (epoch == 50) or (epoch == 100) or (epoch == 200):
        # cut learning rate in half
        lr = lrt.get_value(borrow=False)
        lr = lr / 2.0
        lrt.set_value(floatX(lr))
        b1 = b1t.get_value(borrow=False)
        b1 = b1 + ((0.95 - b1) / 2.0)
        b1t.set_value(floatX(b1))
        lv = lam_vae.get_value(borrow=False)
        lv = lv / 2.0
        lam_vae.set_value(floatX(lv))
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
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    ggn_qtiles = np.percentile(gen_grad_norms, [50., 80., 90., 95.])
    str3 = "    [q50, q80, q90, q95, max](ggn): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            ggn_qtiles[0], ggn_qtiles[1], ggn_qtiles[2], ggn_qtiles[3], np.max(gen_grad_norms))
    ign_qtiles = np.percentile(inf_grad_norms, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](ign): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            ign_qtiles[0], ign_qtiles[1], ign_qtiles[2], ign_qtiles[3], np.max(inf_grad_norms))
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str6 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format( \
            kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str7 = "    module kld -- {}".format(" ".join(kld_strs))
    str8 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, vfe/iwae: {2:.2f}".format( \
            v_epoch_costs[3], v_epoch_costs[4], v_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    # generate some samples from the model prior
    samples = np.asarray(sample_func(sample_z0mb))
    grayscale_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(result_dir, epoch))
    # test reconstruction performance (inference + generation)
    tr_rb = Xtr[0:100,:]
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
