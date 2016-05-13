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
from ModelBuilders import build_mnist_conv_res, build_mnist_conv_res_hires

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_conv_gmm_ss'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load MNIST dataset, either fixed or dynamic binarization
data_path = "{}/data/".format(EXP_DIR)

dataset = load_udm_ss("{}mnist.pkl.gz".format(data_path), sup_count=100)
Xtr_un = dataset['Xtr_un']
Ytr_un = 0. * one_hot(dataset['Ytr_un'], n=10)
Xtr_su = dataset['Xtr_su']
Ytr_su = 1. * one_hot(dataset['Ytr_su'], n=10)
Xva = dataset['Xva']
Yva = 0. * one_hot(dataset['Yva'], n=10)
Xte = dataset['Xte']
Yte = 0. * one_hot(dataset['Yte'], n=10)


set_seed(123)        # seed for shared rngs
nbatch = 100         # # of examples in batch
nc = 1               # # of channels in image
nz0 = 32             # # of dim in top-most latent variables
npx = 28             # # of pixels width/height of images
nx = npx * npx * nc  # # of dimensions in X
niter = 150          # # of iter at starting learning rate
niter_decay = 150    # # of iter to linearly decay learning rate to zero
use_td_cond = False
use_bn = True
mix_comps = 10
depth_7x7 = 5
depth_14x14 = 5

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xtr_un.shape[0]


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
inf_gen_model = \
    build_mnist_conv_res(
        nz0=nz0, nz1=4, ngf=32, ngfc=128, use_bn=use_bn,
        act_func='lrelu', use_td_cond=use_td_cond, mix_comps=mix_comps,
        shared_dim=(nz0 // 2), depth_7x7=depth_7x7, depth_14x14=depth_14x14)
td_modules = inf_gen_model.td_modules
bu_modules = inf_gen_model.bu_modules
im_modules = inf_gen_model.im_modules
mix_module = inf_gen_model.mix_module

# inf_gen_model.load_params(inf_gen_param_file)


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
gen_params = inf_gen_model.gen_params
inf_params = inf_gen_model.inf_params
g_params = gen_params + inf_params

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg_un = T.tensor4()
Xg_su = T.tensor4()
Yg_un = T.matrix()
Yg_su = T.matrix()

Xg = T.concatenate([Xg_un, Xg_su], axis=1)
Yg = T.concatenate([Yg_un, Yg_su], axis=1)
Yg = T.tensor4()
Z0 = T.matrix()

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in g_params])

# run an inference and reconstruction pass through the generative stuff
im_res_dict = inf_gen_model.apply_im(Xg, kl_mode='monte-carlo')
Xg_recon = clip_sigmoid(im_res_dict['td_output'])
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])
mix_comp_kld = im_res_dict['mix_comp_kld']
mix_comp_post = im_res_dict['mix_comp_post']
mix_post_ent = T.mean(im_res_dict['mix_post_ent'])
mix_comp_weight = im_res_dict['mix_comp_weight']

# get mixture component posteriors and KLds for the supervised inputs
Yh_su = mix_comp_post[Xg_un.shape[0]:, :]
mix_comp_kld_su = mix_comp_kld[Xg_un.shape[0]:, :]

# compute various metrics associated with the supervised inputs
cls_nll_su = -T.mean(T.log(T.sum(Yg_su * Yh_su, axis=1)))
acc_su = T.sum(T.cast(T.argmax(Yg_su, axis=1) == T.argmax(Yh_su, axis=1), 'floatX')) / T.sum(Yg_su)
kld_su = T.mean(T.sum(Yg_su * mix_comp_kld_su, axis=1))

# compute reconstruction error part of free-energy
log_p_x = T.sum(log_prob_bernoulli(
                T.flatten(Xg, 2), T.flatten(Xg_recon, 2),
                do_sum=False), axis=1)
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, T.sum(mod_kld, axis=1)) for mod_name, mod_kld in kld_dict.items()]
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
full_cost = vae_nll_cost + opt_kld_cost + vae_reg_cost + (0.1 * cls_nll_su)

# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)
Xd_model = clip_sigmoid(Xd_model)

# run a basic reconstruction pass through the model
Xgr = T.tensor4()
rec_dict = inf_gen_model.apply_im(Xgr, kl_mode='monte-carlo')
Xgr_recon = clip_sigmoid(rec_dict['td_output'])

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
gen_updates, gen_grads = gen_updater(gen_params, full_cost, return_grads=True)
inf_updates, inf_grads = inf_updater(inf_params, full_cost, return_grads=True)
g_updates = gen_updates + inf_updates
gen_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in gen_grads]))
inf_grad_norm = T.sqrt(sum([T.sum(g**2.) for g in inf_grads]))
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xgr], Xgr_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr_un[0:100, :]))
print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost, mix_post_ent, vae_cost, vae_nll_cost,
                 vae_kld_cost, acc_su, cls_nll_su,
                 vae_obs_costs, vae_layer_klds, mix_comp_weight]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost', 'mix_post_ent', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'acc_su', 'cls_nll_su',
              'vae_obs_costs', 'vae_layer_klds', 'mix_comp_weight']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg_un, Yg_un, Xg_su, Yg_su], g_cost_outputs, updates=g_updates)
g_eval_func = theano.function([Xg_un, Yg_un, Xg_su, Yg_su], g_cost_outputs)
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
kld_weights = np.linspace(0.0, 1.0, 10)
for epoch in range(1, (niter + niter_decay + 1)):
    Xtr_un, Ytr_un = shuffle_simultaneously([Xtr_un, Ytr_un])
    Xva, Yva = shuffle_simultaneously([Xva, Yva])
    # mess with the KLd cost
    # if ((epoch-1) < len(kld_weights)):
    #     lam_kld.set_value(floatX([kld_weights[epoch-1]]))
    lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(7)]
    v_epoch_costs = [0. for i in range(7)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    epoch_comp_weights = [0. for i in range(mix_comps)]
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    v_batch_count = 0.
    for imb_x in tqdm(iter_data(Xtr_un, size=nbatch), total=(ntrain / nbatch)):
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count) * nbatch
            vmb_x = Xva[start_idx:(start_idx + nbatch), :]
            vmb_y = Yva[start_idx:(start_idx + nbatch), :]
        else:
            vmb_x = Xva[0:nbatch, :]
            vmb_y = Yva[0:nbatch, :]
        # transform noisy training batch and carry buffer to "image format"
        imb_img = train_transform(imb_x)
        imb_cls = Ytr_un[:imb_img.shape[0], :]
        smb_img = train_transform(Xtr_su)
        smb_cls = Ytr_su
        vmb_img = train_transform(vmb_x)
        vmb_cls = vmb_y
        # train vae on training batch
        g_result = g_train_func(imb_img, imb_cls, smb_img, smb_cls)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:7], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        batch_obs_costs = g_result[7]
        batch_layer_klds = g_result[8]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        batch_comp_weights = g_result[9]
        epoch_comp_weights = [(v1 + v2) for v1, v2 in zip(batch_comp_weights, epoch_comp_weights)]
        g_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            v_result = g_eval_func(vmb_img, 0. * vmb_cls, vmb_img, vmb_cls)
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
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    epoch_comp_weights = [(c / g_batch_count) for c in epoch_comp_weights]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:7], g_bc_names[:7])]
    str2 = " ".join(g_bc_strs)
    v_bc_strs = ["{0:s}: {1:.2f},".format(c_name, v_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:7], g_bc_names[:7])]
    str3 = " ".join(v_bc_strs)
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    mcw_qtiles = np.percentile(epoch_comp_weights, [20., 40., 60., 80.])
    str6 = "    [q20, q40, q60, q80, max](mcw): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        mcw_qtiles[0], mcw_qtiles[1], mcw_qtiles[2], mcw_qtiles[3], np.max(epoch_comp_weights))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str7 = "    module kld -- {}".format(" ".join(kld_strs))
    str8 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, vfe/iwae: {2:.2f}".format(
        v_epoch_costs[3], v_epoch_costs[4], v_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
    #################################
    # QUALITATIVE DIAGNOSTICS STUFF #
    #################################
    if (epoch < 20) or (((epoch - 1) % 20) == 0):
        # generate some samples from the model prior
        comp_idx = np.arange(mix_comps).repeat(10)
        sample_z0mb = mix_module.sample_mix_comps(comp_idx)
        samples = np.asarray(sample_func(sample_z0mb))
        grayscale_grid_vis(draw_transform(samples), (mix_comps, 10),
                           "{}/gen_{}.png".format(result_dir, epoch))
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
