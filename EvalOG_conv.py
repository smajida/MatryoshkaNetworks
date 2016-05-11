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
desc = 'test_conv_5deep'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load MNIST dataset, either fixed or dynamic binarization
data_path = "{}/data/".format(EXP_DIR)
Xtr, Ytr, Xva, Yva = load_omniglot(data_path, target_type='one-hot')

set_seed(123)        # seed for shared rngs
nbatch = 10         # # of examples in batch
nc = 1               # # of channels in image
nz0 = 32             # # of dim in top-most latent variables
nz1 = 4              # # of dim in intermediate latent variables
ngf = 32             # base # of filters for conv layers
ngfc = 256           # # of dim in top-most hidden layer
npx = 28             # # of pixels width/height of images
nx = npx * npx * nc  # # of dimensions in X
niter = 200          # # of iter at starting learning rate
niter_decay = 200    # # of iter to linearly decay learning rate to zero
iwae_samples = 10
use_td_cond = False
depth_7x7 = 5
depth_14x14 = 5
depth_28x28 = None

fine_tune_inf_net = False

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xva.shape[0]


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output


def np_log_mean_exp(x, axis=None):
    assert (axis is not None), "please provide an axis..."
    m = np.max(x, axis=axis, keepdims=True)
    lme = m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))
    return lme


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


###################
# BUILD THE MODEL #
###################

if depth_28x28 is None:
    inf_gen_model = \
        build_og_conv_res(
            nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc, use_bn=False,
            act_func='lrelu', use_td_cond=use_td_cond,
            depth_7x7=depth_7x7, depth_14x14=depth_14x14)
else:
    inf_gen_model = \
        build_og_conv_res_hires(
            nz0=nz0, nz1=nz1, ngf=ngf, ngfc=ngfc, use_bn=False,
            act_func='lrelu', use_td_cond=use_td_cond,
            depth_7x7=depth_7x7, depth_14x14=depth_14x14, depth_28x28=depth_28x28)
td_modules = inf_gen_model.td_modules
bu_modules = inf_gen_model.bu_modules
im_modules = inf_gen_model.im_modules


###################
# LOAD PARAMETERS #
###################
inf_gen_model.load_params(inf_gen_param_file)


# FINE TUNE THE INFERENCE NETWORK ON TEST SET
if fine_tune_inf_net:
    ####################################
    # Setup the optimization objective #
    ####################################
    lam_kld = sharedX(floatX([1.0]))
    inf_params = inf_gen_model.inf_params

    ##########################################################
    # CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
    ##########################################################
    Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
    # parameter regularization part of cost
    vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in inf_params])

    # run an inference and reconstruction pass through the generative stuff
    im_res_dict = inf_gen_model.apply_im(Xg)
    Xg_recon = clip_sigmoid(im_res_dict['td_output'])
    kld_dict = im_res_dict['kld_dict']
    log_p_z = sum(im_res_dict['log_p_z'])
    log_q_z = sum(im_res_dict['log_q_z'])

    log_p_x = T.sum(log_prob_bernoulli(
                    T.flatten(Xg, 2), T.flatten(Xg_recon, 2),
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

    #################################################################
    # COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
    #################################################################

    # stuff for performing updates
    lrt = sharedX(0.0001)
    b1t = sharedX(0.8)
    inf_updater = updates.Adam(lr=lrt, b1=b1t, b2=0.98, e=1e-4, clipnorm=1000.0)

    # build training cost and update functions
    t = time()
    print("Computing gradients...")
    inf_updates, inf_grads = inf_updater(inf_params, full_cost_inf, return_grads=True)
    print("Compiling training functions...")
    # collect costs for generator parameters
    g_basic_costs = [full_cost_gen, full_cost_inf, vae_cost, vae_nll_cost,
                     vae_kld_cost]
    g_bc_idx = range(0, len(g_basic_costs))
    g_bc_names = ['full_cost_gen', 'full_cost_inf', 'vae_cost', 'vae_nll_cost',
                  'vae_kld_cost']
    g_cost_outputs = g_basic_costs
    # compile function for computing generator costs and updates
    i_train_func = theano.function([Xg], g_cost_outputs, updates=inf_updates)
    print "{0:.2f} seconds to compile theano functions".format(time() - t)

    # make file for recording test progress
    log_name = "{}/FINE-TUNE.txt".format(result_dir)
    out_file = open(log_name, 'wb')

    print("EXPERIMENT: {}".format(desc.upper()))
    n_check = 0
    n_updates = 0
    t = time()
    for epoch in range(1, 200):
        Xva = shuffle(Xva)
        # initialize cost arrays
        g_epoch_costs = [0. for gco in g_cost_outputs]
        g_batch_count = 0.
        if (epoch < 25):
            lrt.set_value(floatX(0.00001))
        elif (epoch < 50):
            lrt.set_value(floatX(0.00003))
        for imb in tqdm(iter_data(Xva, size=100), total=(ntrain / 100)):
            # transform training batch to "image format"
            imb_img = train_transform(imb)
            # train vae on training batch
            g_result = i_train_func(floatX(imb_img))
            g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
            g_batch_count += 1
        if (epoch == 75) or (epoch == 150):
            lr = lrt.get_value(borrow=False)
            lr = lr / 2.0
            lrt.set_value(floatX(lr))
        # report quantitative diagnostics
        g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
        str1 = "Epoch {}: ({})".format(epoch, desc.upper())
        g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                     for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
        str2 = " ".join(g_bc_strs)
        joint_str = "\n".join([str1, str2])
        print(joint_str)
        out_file.write(joint_str + "\n")
        out_file.flush()


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
Xg_rep_recon = clip_sigmoid(im_res_dict['td_output'])
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

log_p_x = T.sum(log_prob_bernoulli(
                T.flatten(Xg_rep, 2), T.flatten(Xg_rep_recon, 2),
                do_sum=False), axis=1)

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

######################################################
# Get functions for free sampling and reconstruction #
######################################################
# get simple reconstruction, for other purposes
im_rd = inf_gen_model.apply_im(Xg)
Xg_recon = clip_sigmoid(im_rd['td_output'])
# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)
Xd_model = clip_sigmoid(Xd_model)


# build training cost and update functions
t = time()
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr[0:100, :]))
print("Compiling cost computing functions...")
# collect costs for generator parameters
g_basic_costs = [iwae_bound, vae_bound, vae_nll_cost, vae_kld_cost,
                 iwae_bound_lme]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['iwae_bound', 'vae_bound', 'vae_nll_cost', 'vae_kld_cost',
              'iwae_bound_lme']
# compile function for computing generator costs and updates
iwae_cost_func = theano.function([Xg], [log_p_x, log_p_z, log_q_z])
g_eval_func = theano.function([Xg], g_basic_costs)
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
            # evaluate costs more thoroughly
            iwae_bounds = iwae_multi_eval(imb_img, 25,
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
        ######################
        # DRAW SOME PICTURES #
        ######################
        sample_z0mb = np.repeat(rand_gen(size=(20, nz0)), 20, axis=0)
        samples = np.asarray(sample_func(sample_z0mb))
        grayscale_grid_vis(draw_transform(samples), (20, 20), "{}/eval_gen_e{}_b{}.png".format(result_dir, epoch, block_num))
    epoch_vae_cost = epoch_vae_cost / len(Xva_blocks)
    epoch_iwae_cost = epoch_iwae_cost / len(Xva_blocks)
    str1 = "EPOCH {0:d} -- vae: {1:.2f}, iwae: {2:.2f}".format(epoch, epoch_vae_cost, epoch_iwae_cost)
    print(str1)
    out_file.write(str1 + "\n")
    out_file.flush()






##############
# EYE BUFFER #
##############
