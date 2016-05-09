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
from lib.data_utils import shuffle, iter_data, shuffle_simultaneously
from load import load_udm_ss, one_hot

#
# Phil's business
#
from ModelBuilders import build_mnist_conv_res, build_mnist_conv_res_ss

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_conv_ss_100'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load MNIST dataset semi-supervised data
data_path = "{}/data/".format(EXP_DIR)

data_dict = load_udm_ss("{}mnist.pkl.gz".format(data_path), 100)
Xtr_su = data_dict['Xtr_su']
Ytr_su = floatX(one_hot(data_dict['Ytr_su'], 10))
Xtr_un = data_dict['Xtr_un']
Ytr_un = floatX(one_hot(data_dict['Ytr_un'], 10))
Xva = data_dict['Xva']
Yva = floatX(one_hot(data_dict['Yva'], 10))
Xte = data_dict['Xte']
Yte = floatX(one_hot(data_dict['Yte'], 10))

# stack the supervised examples into the unsupervised set
Xtr_un = np.concatenate([Xtr_su, Xtr_un], axis=0)
Ytr_un = np.concatenate([Ytr_su, Ytr_un], axis=0)

print('Xtr_su.shape: {}'.format(Xtr_su.shape))
print('Ytr_su.shape: {}'.format(Ytr_su.shape))
print('Xtr_un.shape: {}'.format(Xtr_un.shape))
print('Ytr_un.shape: {}'.format(Ytr_un.shape))

set_seed(123)        # seed for shared rngs
nbatch = 100         # # of examples in batch
nc = 1               # # of channels in image
nz0 = 32             # # of dim in top-most latent variables
npx = 28             # # of pixels width/height of images
nx = npx * npx * nc  # # of dimensions in X
niter = 150          # # of iter at starting learning rate
niter_decay = 150    # # of iter to linearly decay learning rate to zero
use_bn = True
use_td_cond = False
depth_7x7 = 1
depth_14x14 = 1

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xtr_un.shape[0]


def train_transform(X):
    # binarize data and reshape for input to convnet
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
softmax = T.nnet.softmax

# BUILD THE MODEL
inf_gen_model = \
    build_mnist_conv_res_ss(
        nz0=nz0, nz1=4, ngf=32, ngfc=128, class_count=10,
        act_func='lrelu', use_bn=True, use_td_cond=True,
        depth_7x7=depth_7x7, depth_14x14=depth_14x14)
td_modules = inf_gen_model.td_modules
bu_modules = inf_gen_model.bu_modules
im_modules = inf_gen_model.im_modules
cls_module = inf_gen_model.cls_module

# inf_gen_model.load_params(inf_gen_param_file)


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
lam_su_vae = sharedX(floatX([0.01]))
lam_su_cls = sharedX(floatX([0.01]))
gen_params = inf_gen_model.gen_params
inf_params = inf_gen_model.inf_params
g_params = gen_params + inf_params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg_un = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Xg_su = T.tensor4()  # symbolic var for inputs to bottom-up inference network
Yg_su = T.matrix()   # symbolic var for class label inputs
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in g_params])

# run an inference and reconstruction pass through the generative stuff
Xg = T.concatenate([Xg_un, Xg_su], axis=0)
im_res_dict = inf_gen_model.apply_im(Xg)
Xg_recon = clip_sigmoid(im_res_dict['td_output'])
cls_acts = im_res_dict['cls_acts']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

# get observation reconstruction error for all samples
log_p_x = T.sum(log_prob_bernoulli(
                T.flatten(Xg, 2), T.flatten(Xg_recon, 2),
                do_sum=False), axis=1)

# get class prediction error for supervised samples
p_y_su = softmax(cls_acts[Xg_un.shape[0]:, :])
log_p_y_su = T.sum((Yg_su * T.log(p_y_su)), axis=1)  # Yg_su must be one-hot

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_obs_nlls_un = vae_obs_nlls[:Xg_un.shape[0]]
vae_obs_nlls_su = vae_obs_nlls[Xg_un.shape[0]:]
su_nll_cost = T.mean(vae_obs_nlls_su)
su_cls_cost = -T.mean(log_p_y_su)
#
nll_cost_un = T.mean(vae_obs_nlls_un)
nll_cost_su = lam_su_vae[0] * su_nll_cost + lam_su_cls[0] * su_cls_cost


# combine unsupervised and supervised reconstruction costs
vae_nll_cost = nll_cost_un + nll_cost_su

# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dict.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]

# split the KLd cost into unsupervised and supervised parts
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
vae_obs_klds_un = vae_obs_klds[:Xg_un.shape[0]]
vae_obs_klds_su = vae_obs_klds[Xg_un.shape[0]:]
vae_kld_cost = T.mean(vae_obs_klds_un) + (lam_su_vae[0] * T.mean(vae_obs_klds_su))

# compute the KLd cost to use for optimization
opt_kld_cost = lam_kld[0] * vae_kld_cost

# combined cost for generator stuff
vae_obs_costs = vae_obs_nlls_un + vae_obs_klds_un
vae_cost = T.mean(vae_obs_costs)

# cost used by the optimizer
full_cost_gen = vae_nll_cost + opt_kld_cost + vae_reg_cost
full_cost_inf = full_cost_gen

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

Xr = T.tensor4()
xr_res_dict = inf_gen_model.apply_im(Xr)
Xr_recon = clip_sigmoid(xr_res_dict['td_output'])

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
recon_func = theano.function([Xr], Xr_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr_un[0:100, :]))
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost_gen, full_cost_inf, vae_cost, vae_nll_cost,
                 vae_kld_cost, su_nll_cost, su_cls_cost,
                 vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost_gen', 'full_cost_inf', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'su_nll_cost', 'su_cls_cost',
              'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg_un, Xg_su, Yg_su], g_cost_outputs, updates=g_updates)
g_eval_func = theano.function([Xg_un, Xg_su, Yg_su], g_cost_outputs)
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
kld_weights = np.linspace(0.0, 1.0, 10)
sample_z0mb = rand_gen(size=(200, nz0))
for epoch in range(1, (niter + niter_decay + 1)):
    Xtr_un = shuffle(Xtr_un)
    Xva, Yva = shuffle_simultaneously([Xva, Yva])
    # mess with the KLd cost
    # if ((epoch-1) < len(kld_weights)):
    #     lam_kld.set_value(floatX([kld_weights[epoch-1]]))
    lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(7)]
    v_epoch_costs = [0. for i in range(7)]
    i_epoch_costs = [0. for i in range(7)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    i_batch_count = 0.
    v_batch_count = 0.
    for imb in tqdm(iter_data(Xtr_un, size=nbatch), total=(ntrain / nbatch)):
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count) * nbatch
            vmb_x = Xva[start_idx:(start_idx + nbatch), :]
            vmb_y = Yva[start_idx:(start_idx + nbatch), :]
        else:
            vmb_x = Xva[0:nbatch, :]
            vmb_y = Yva[0:nbatch, :]
        # transform noisy training batch and carry buffer to "image format"
        imb_img = train_transform(imb)
        vmb_img = train_transform(vmb_x)
        xsu_img = train_transform(Xtr_su)
        # train vae on training batch
        g_result = g_train_func(imb_img, xsu_img, Ytr_su)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:7], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        batch_obs_costs = g_result[7]
        batch_layer_klds = g_result[8]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # train inference model on samples from the generator
        i_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            v_result = g_eval_func(vmb_img, vmb_img, vmb_y)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:7], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 5) or (epoch == 15) or (epoch == 30) or (epoch == 60) or (epoch == 100):
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
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    i_epoch_costs = [(c / i_batch_count) for c in i_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:7], g_bc_names[:7])]
    str2 = " ".join(g_bc_strs)
    i_bc_strs = ["{0:s}: {1:.2f},".format(c_name, i_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:7], g_bc_names[:7])]
    str2i = " ".join(i_bc_strs)
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str6 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str7 = "    module kld -- {}".format(" ".join(kld_strs))
    str8 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, cls: {2:.2f}".format(
        v_epoch_costs[3], v_epoch_costs[4], v_epoch_costs[6])
    joint_str = "\n".join([str1, str2, str2i, str5, str6, str7, str8])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    if (epoch < 20) or (((epoch - 1) % 20) == 0):
        # generate some samples from the model prior
        samples = np.asarray(sample_func(sample_z0mb))
        grayscale_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(result_dir, epoch))
        # test reconstruction performance (inference + generation)
        tr_rb = Xtr_un[0:100, :]
        va_rb = Xva[0:100, :]
        # get the model reconstructions
        tr_rb = train_transform(tr_rb)
        va_rb = train_transform(va_rb)
        tr_recons = recon_func(tr_rb)
        va_recons = recon_func(va_rb)
        # stripe data for nice display (each reconstruction next to its target)
        tr_vis_batch = np.zeros((200, 1, npx, npx))
        va_vis_batch = np.zeros((200, 1, npx, npx))
        for rec_pair in range(100):
            idx_in = 2 * rec_pair
            idx_out = 2 * rec_pair + 1
            tr_vis_batch[idx_in, :, :, :] = tr_rb[rec_pair, :, :, :]
            tr_vis_batch[idx_out, :, :, :] = tr_recons[rec_pair, :, :, :]
            va_vis_batch[idx_in, :, :, :] = va_rb[rec_pair, :, :, :]
            va_vis_batch[idx_out, :, :, :] = va_recons[rec_pair, :, :, :]
        # draw images...
        grayscale_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(result_dir, epoch))
        grayscale_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(result_dir, epoch))








##############
# EYE BUFFER #
##############
