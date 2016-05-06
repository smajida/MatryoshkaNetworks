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
from lib.data_utils import \
    shuffle, iter_data, get_masked_data, get_downsampling_masks
from load import load_binarized_mnist, load_udm

#
# Phil's business
#
from MatryoshkaModulesRNN import \
    GenFCGRUModuleRNN, FCReshapeModuleRNN, TDModuleWrapperRNN, \
    GenConvGRUModuleRNN, BasicConvModuleRNN, \
    InfFCGRUModuleRNN, InfConvGRUModuleRNN
from MatryoshkaNetworksRNN import DeepSeqCondGen

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_uncond_conv_gru_1deep'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fixed_binarization = True
# load MNIST dataset, either fixed or dynamic binarization
data_path = "{}/data/".format(EXP_DIR)
if fixed_binarization:
    Xtr, Xva, Xte = load_binarized_mnist(data_path=data_path)
    Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
    Xva = Xte
    Xmu = np.mean(Xtr, axis=0)
else:
    dataset = load_udm("{}mnist.pkl.gz".format(data_path), to_01=True)
    Xtr = dataset[0][0]
    Xva = dataset[1][0]
    Xte = dataset[2][0]
    Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
    Xva = Xte
    Xmu = np.mean(Xtr, axis=0)


set_seed(123)      # seed for shared rngs
nc = 1             # # of channels in image
nbatch = 100       # # of examples in batch
npx = 28           # # of pixels width/height of images
nz0 = 64           # # of dim for Z0
nz1 = 8            # # of dim for Z1
ngf = 32           # base # of channels for defining layers
nx = npx * npx * nc  # # of dimensions in X
niter = 150        # # of iter at starting learning rate
niter_decay = 250  # # of iter to linearly decay learning rate to zero
td_act_func = 'lrelu'  # activation function for top-down modules
bu_act_func = 'lrelu'  # activation function for bottom-up modules
use_td_cond = True

ntrain = Xtr.shape[0]


def train_transform(X):
    # transform vectorized observations into convnet inputs
    if not fixed_binarization:
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

#########################################
# Setup the top-down processing modules #
# -- these do generation                #
#########################################

# setup the (28, 28) -> (28, 28) module
td_module_1a = \
    GenConvGRUModuleRNN(
        state_chans=(ngf * 2),
        input_chans=(ngf * 2),
        rand_chans=nz1,
        spatial_shape=(28, 28),
        filt_shape=(3, 3),
        act_func='tanh',
        mod_name='td_mod_1a')
td_module_1b = \
    BasicConvModuleRNN(
        in_chans=(ngf * 2),
        out_chans=nc,
        filt_shape=(5, 5),
        stride='single',
        act_func='ident',
        mod_name='td_mod_1b')
td_module_1 = \
    TDModuleWrapperRNN(
        gen_module=td_module_1a,
        mlp_modules=[td_module_1b],
        mod_name='td_mod_1')

td_modules = [td_module_1]

##########################################
# Setup the bottom-up processing modules #
# -- these do generation inference       #
##########################################

bu_module_1 = \
    BasicConvModuleRNN(
        in_chans=(3 * nc),
        out_chans=(ngf * 2),
        filt_shape=(5, 5),
        stride='single',
        act_func='ident',
        mod_name='bu_mod_1')  # (28, 28) -> (28, 28)

# modules must be listed in "evaluation order"
bu_modules_gen = [bu_module_1]


##########################################
# Setup the bottom-up processing modules #
# -- these do generation inference       #
##########################################

bu_module_1 = \
    BasicConvModuleRNN(
        in_chans=(3 * nc),
        out_chans=(ngf * 2),
        filt_shape=(5, 5),
        stride='single',
        act_func='ident',
        mod_name='bu_mod_1')  # (28, 28) -> (28, 28)

# modules must be listed in "evaluation order"
bu_modules_inf = [bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_1 = \
    InfConvGRUModuleRNN(
        state_chans=(ngf * 2),
        td_state_chans=(ngf * 2),
        td_input_chans=(ngf * 2),
        bu_chans=(ngf * 2),
        rand_chans=nz1,
        spatial_shape=(28, 28),
        act_func='tanh',
        use_td_cond=use_td_cond,
        mod_name='im_mod_1')

im_modules_gen = [im_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_1 = \
    InfConvGRUModuleRNN(
        state_chans=(ngf * 2),
        td_state_chans=(ngf * 2),
        td_input_chans=(ngf * 2),
        bu_chans=(ngf * 2),
        rand_chans=nz1,
        spatial_shape=(28, 28),
        act_func='tanh',
        use_td_cond=use_td_cond,
        mod_name='im_mod_1')

im_modules_inf = [im_module_1]

#
# Setup a description for where to get conditional distributions from.
#
merge_info = {
    'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                 'bu_module': 'bu_mod_1'}
}


# construct the "wrapper" object for managing all our modules
seq_cond_gen_model = \
    DeepSeqCondGen(
        td_modules=td_modules,
        bu_modules_gen=bu_modules_gen,
        im_modules_gen=im_modules_gen,
        bu_modules_inf=bu_modules_inf,
        im_modules_inf=im_modules_inf,
        merge_info=merge_info)

# inf_gen_model.load_params(inf_gen_param_file)

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
c0 = sharedX(floatX(np.zeros((1, nc, npx, npx))))
gen_params = seq_cond_gen_model.gen_params + [c0]
inf_params = seq_cond_gen_model.inf_params
all_params = seq_cond_gen_model.all_params + [c0]


######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output


def make_model_input(x_in):
    xg_inf = train_transform(x_in)
    return [xg_inf]

#
# test the model implementation
#
x_in = T.tensor4()
c0_in = T.repeat(c0, x_in.shape[0], axis=0)
xa_inf = T.concatenate([x_in, c0_in, (x_in - c0_in)], axis=1)
res_dict = \
    seq_cond_gen_model.apply_im_uncond(
        input_inf=xa_inf,
        td_states=None,
        im_states_inf=None)
x_recon = res_dict['output']

print('Compiling test function...')
inputs = [x_in]
outputs = [x_recon]
test_func = theano.function(inputs, outputs)

model_input = make_model_input(Xtr[0:100, :])
test_out = test_func(*model_input)
print('DONE.')


##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################

# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in all_params])

# feed all masked inputs through the inference network
x_in = T.tensor4()
td_states = None
im_states_inf = None
canvas = T.repeat(c0, x_in.shape[0], axis=0)
kld_dicts = []
step_recons = []
for i in range(10):
    # record initial canvas state for each step
    xg_gen = clip_sigmoid(canvas)
    step_recons.append(xg_gen)
    # perform pass through joint inference/generation model
    xa_inf = T.concatenate([x_in, xg_gen, (x_in - xg_gen)], axis=1)
    res_dict = \
        seq_cond_gen_model.apply_im_uncond(
            input_inf=xa_inf,
            td_states=td_states,
            im_states_inf=im_states_inf)
    # update canvas state
    canvas = canvas + res_dict['output']
    # grab updated states for next refinement step
    td_states = res_dict['td_states']
    im_states_inf = res_dict['im_states_inf']
    # record klds from this step
    kld_dicts.append(res_dict['kld_dict'])
# reconstruction uses canvas after final refinement step
xg_recon = sigmoid(T.clip(canvas, -15., 15.))
step_recons.append(xg_recon)

# compute masked reconstruction error from final step.
log_p_x = T.sum(log_prob_bernoulli(
                T.flatten(x_in, 2), T.flatten(xg_recon, 2),
                mask=None, do_sum=False),
                axis=1)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# convert KL dict to aggregate KLds over inference steps
kl_by_td_mod = {tdm_name: sum([kld_dict[tdm_name] for kld_dict in kld_dicts])
                for tdm_name in kld_dicts[0].keys()}  # aggregate over refinement steps
# kl_by_td_mod = {tdm_name: T.sum(kl_by_td_mod[tdm_name], axis=1)
#                 for tdm_name in kld_dicts[0].keys()}  # aggregate over latent dimensions
# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, mod_kld) for mod_name, mod_kld in kl_by_td_mod.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]

# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

# compute per-layer KL-divergence part of cost
alt_layer_klds = [mod_kld**2.0 for mod_name, mod_kld in kld_tuples]
alt_kld_cost = T.mean(sum(alt_layer_klds))

# compute the KLd cost to use for optimization
opt_kld_cost = (lam_kld[0] * vae_kld_cost) + ((1.0 - lam_kld[0]) * alt_kld_cost)

# combined cost for generator stuff
vae_cost = vae_nll_cost + vae_kld_cost
vae_obs_costs = vae_obs_nlls + vae_obs_klds
# cost used by the optimizer
full_cost = vae_nll_cost + opt_kld_cost + vae_reg_cost


#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

# stuff for performing updates
lrt = sharedX(0.0005)
b1t = sharedX(0.9)
updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=1000.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
all_updates, all_grads = updater(all_params, full_cost, return_grads=True)

print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([x_in], step_recons)
model_input = make_model_input(Xtr[0:100, :])
test_recons = recon_func(*model_input)

print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost, full_cost, vae_cost, vae_nll_cost,
                 vae_kld_cost, vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost', 'full_cost', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([x_in], g_cost_outputs, updates=all_updates)
g_eval_func = theano.function([x_in], g_cost_outputs)
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
    Xtr = shuffle(Xtr)
    Xva = shuffle(Xva)
    # mess with the KLd cost
    # if ((epoch-1) < len(kld_weights)):
    #     lam_kld.set_value(floatX([kld_weights[epoch-1]]))
    lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    i_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    i_batch_count = 0.
    v_batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=(ntrain / nbatch)):
        # grab a validation batch, if required
        if v_batch_count < 50:
            start_idx = int(v_batch_count) * nbatch
            vmb = Xva[start_idx:(start_idx + nbatch), :]
        else:
            vmb = Xva[0:nbatch, :]
        # transform training batch validation batch to model input format
        imb_input = make_model_input(imb)
        vmb_input = make_model_input(vmb)
        # train vae on training batch
        g_result = g_train_func(*imb_input)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        batch_obs_costs = g_result[5]
        batch_layer_klds = g_result[6]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # train inference model on samples from the generator
        # if epoch > 5:
        #     smb_img = binarize_data(sample_func(rand_gen(size=(100, nz0))))
        #     i_result = i_train_func(smb_img)
        #     i_epoch_costs = [(v1 + v2) for v1, v2 in zip(i_result[:5], i_epoch_costs)]
        i_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            v_result = g_eval_func(*vmb_input)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:5], v_epoch_costs)]
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
    seq_cond_gen_model.dump_params(inf_gen_param_file)
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    i_epoch_costs = [(c / i_batch_count) for c in i_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    i_bc_strs = ["{0:s}: {1:.2f},".format(c_name, i_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str3 = " ".join(i_bc_strs)
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str5 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str6 = "    module kld -- {}".format(" ".join(kld_strs))
    str7 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, vfe/iwae: {2:.2f}".format(
        v_epoch_costs[3], v_epoch_costs[4], v_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
#     #################################
#     # QUALITATIVE DIAGNOSTICS STUFF #
#     #################################
#     if (epoch < 20) or (((epoch - 1) % 20) == 0):
#         # generate some samples from the model prior
#         samples = np.asarray(sample_func(sample_z0mb))
#         grayscale_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(result_dir, epoch))
#         # test reconstruction performance (inference + generation)
#         tr_rb = Xtr[0:100, :]
#         va_rb = Xva[0:100, :]
#         # get the model reconstructions
#         tr_rb = train_transform(tr_rb)
#         va_rb = train_transform(va_rb)
#         tr_recons = recon_func(tr_rb)
#         va_recons = recon_func(va_rb)
#         # stripe data for nice display (each reconstruction next to its target)
#         tr_vis_batch = np.zeros((200, nc, npx, npx))
#         va_vis_batch = np.zeros((200, nc, npx, npx))
#         for rec_pair in range(100):
#             idx_in = 2 * rec_pair
#             idx_out = 2 * rec_pair + 1
#             tr_vis_batch[idx_in, :, :, :] = tr_rb[rec_pair, :, :, :]
#             tr_vis_batch[idx_out, :, :, :] = tr_recons[rec_pair, :, :, :]
#             va_vis_batch[idx_in, :, :, :] = va_rb[rec_pair, :, :, :]
#             va_vis_batch[idx_out, :, :, :] = va_recons[rec_pair, :, :, :]
#         # draw images...
#         grayscale_grid_vis(draw_transform(tr_vis_batch), (10, 20), "{}/rec_tr_{}.png".format(result_dir, epoch))
#         grayscale_grid_vis(draw_transform(va_vis_batch), (10, 20), "{}/rec_va_{}.png".format(result_dir, epoch))
#
#
#
#
#


##############
# EYE BUFFER #
##############
