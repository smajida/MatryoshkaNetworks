import os
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm

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
from lib.ops import log_mean_exp, binarize_data
from lib.costs import log_prob_bernoulli
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import load_binarized_mnist, load_udm

#
# Phil's business
#
from MatryoshkaModules import BasicFCModule, GenFCResModule, \
                              GenTopModule, InfFCMergeModule, \
                              InfTopModule, BasicFCResModule
from MatryoshkaNetworks import InfGenModel

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_fc_all_noise_010_dyn_bin_old_sc_deeper'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
inf_gen_param_file = "{}/inf_gen_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

fixed_binarization = False
# load MNIST dataset, either fixed or dynamic binarization
data_path = "{}/data/".format(EXP_DIR)
if fixed_binarization:
    Xtr, Xva, Xte = load_binarized_mnist(data_path=data_path)
    Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
    Xva = Xte
else:
    dataset = load_udm("{}mnist.pkl.gz".format(data_path), to_01=True)
    Xtr = dataset[0][0]
    Xva = dataset[1][0]
    Xte = dataset[2][0]
    Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
    Xva = Xte


set_seed(1)       # seed for shared rngs
nc = 1            # # of channels in image
nbatch = 1000      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 64          # # of dim for Z0
nz1 = 64          # # of dim for Z1
ngf = 64          # base # of filters for conv layers in generative stuff
scf = 256          # base # of filters for conv layers in generative stuff
ngfc = 64         # number of filters in top-most fc layer
nx = npx*npx*nc   # # of dimensions in X
niter = 500       # # of iter at starting learning rate
niter_decay = 1500 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_fc = True     # whether to use "internal" conv layers in gen/disc networks
use_bn = True     # whether to use batch normalization throughout the model
act_func = 'lrelu' # activation func to use where they can be selected
iwae_samples = 4  # number of samples to use in MEN bound
noise_std = 0.0   # amount of noise to inject in BU and IM modules
use_td_noise = False # whether to use noise in TD pass
use_bu_noise = False # whether to use noise in BU pass

ntrain = Xtr.shape[0]


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
    if not fixed_binarization:
        X = binarize_data(X)
    return floatX(X)

def draw_transform(X):
    # transform vectorized observations into drawable greyscale images
    X = X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3)
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

td_module_1 = \
GenTopModule(
    rand_dim=nz0,
    out_shape=(ngf*8,),
    fc_dim=ngfc,
    use_fc=True,
    use_sc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_1'
)

td_module_2 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_2'
)

td_module_3 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_3'
)

td_module_4 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_4'
)

td_module_5 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_5'
)

td_module_6 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_6'
)

td_module_7 = \
GenFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_7'
)

td_module_8 = \
BasicFCModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_8'
)

td_module_9 = \
BasicFCModule(
    in_chans=(ngf*8),
    out_chans=nx,
    apply_bn=False,
    use_noise=False,
    act_func='ident',
    mod_name='td_mod_9'
)

# modules must be listed in "evaluation order"
td_modules = [td_module_1, td_module_2, td_module_3, td_module_4, td_module_5,
              td_module_6, td_module_7, td_module_8, td_module_9]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

bu_module_1 = \
InfTopModule(
    bu_chans=(ngf*8 + scf),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    use_sc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
)

bu_module_2 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_2'
)

bu_module_3 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_3'
)

bu_module_4 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_4'
)

bu_module_5 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_5'
)

bu_module_6 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_6'
)

bu_module_7 = \
BasicFCResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    fc_chans=(ngf*8),
    use_fc=use_fc,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_7'
)

bu_module_8 = \
BasicFCModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_8'
)

bu_module_9 = \
BasicFCModule(
    in_chans=nx,
    out_chans=(ngf*8),
    apply_bn=False,
    act_func=act_func,
    mod_name='bu_mod_9'
)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_9, bu_module_8, bu_module_7, bu_module_6, bu_module_5,
              bu_module_4, bu_module_3, bu_module_2, bu_module_1]


#########################################
# Setup the information merging modules #
#########################################

im_module_2 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_2'
)

im_module_3 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_3'
)

im_module_4 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_4'
)

im_module_5 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_5'
)

im_module_6 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_6'
)

im_module_7 = \
InfFCMergeModule(
    td_chans=(ngf*8),
    bu_chans=(ngf*8 + scf),
    fc_chans=(ngf*8),
    rand_chans=nz1,
    use_fc=True,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_7'
)

im_modules = [im_module_2, im_module_3, im_module_4, im_module_5,
              im_module_6, im_module_7]

#
# Setup a description for where to get conditional distributions from.
#
merge_info = {
    'td_mod_1': {'td_type': 'top', 'im_module': None,
                 'bu_source': 'bu_mod_1', 'im_source': None},

    'td_mod_2': {'td_type': 'cond', 'im_module': 'im_mod_2',
                 'bu_source': 'bu_mod_3', 'im_source': None},

    'td_mod_3': {'td_type': 'cond', 'im_module': 'im_mod_3',
                 'bu_source': 'bu_mod_4', 'im_source': None},

    'td_mod_4': {'td_type': 'cond', 'im_module': 'im_mod_4',
                 'bu_source': 'bu_mod_5', 'im_source': None},

    'td_mod_5': {'td_type': 'cond', 'im_module': 'im_mod_5',
                 'bu_source': 'bu_mod_6', 'im_source': None},

    'td_mod_6': {'td_type': 'cond', 'im_module': 'im_mod_6',
                 'bu_source': 'bu_mod_7', 'im_source': None},

    'td_mod_7': {'td_type': 'cond', 'im_module': 'im_mod_7',
                 'bu_source': 'bu_mod_8', 'im_source': None},

    'td_mod_8': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None},

    'td_mod_9': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None}
}

####################
# Shortcut modules #
####################
sc_module_1 = \
BasicFCModule(
    in_chans=(scf*2),
    out_chans=scf,
    apply_bn=False,
    act_func='ident',
    mod_name='sc_mod_1'
)

sc_module_2 = \
BasicFCModule(
    in_chans=nx,
    out_chans=(scf*2),
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='sc_mod_2'
)

sc_modules = [sc_module_2, sc_module_1]

# construct the "wrapper" object for managing all our modules
output_transform = lambda x: sigmoid(T.clip(x, -15.0, 15.0))
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    sc_modules=sc_modules,
    merge_info=merge_info,
    output_transform=output_transform,
    use_td_noise=use_td_noise,
    use_bu_noise=use_bu_noise
)

###################
# LOAD PARAMETERS #
###################
inf_gen_model.load_params(inf_gen_param_file)

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg = T.matrix()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

######################
# Compute IWAE bound #
######################
# run an inference and reconstruction pass through the generative stuff
batch_size = Xg.shape[0]
Xg_rep = T.extra_ops.repeat(Xg, iwae_samples, axis=0)
im_res_dict = inf_gen_model.apply_im(Xg_rep)
Xg_rep_recon = im_res_dict['td_output']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z']) # per-sample sum of log_p_z
log_q_z = sum(im_res_dict['log_q_z']) # per-sample sum of log_q_z

log_p_x = T.sum(log_prob_bernoulli( \
                T.flatten(Xg_rep,2), T.flatten(Xg_rep_recon,2),
                do_sum=False), axis=1) # per-sample log_p_x

# compute quantities used in the IWAE bound
log_ws_vec = log_p_x + log_p_z - log_q_z
log_ws_mat = log_ws_vec.reshape((batch_size, iwae_samples))
ws_mat = log_ws_mat - T.max(log_ws_mat, axis=1, keepdims=True)
ws_mat = T.exp(ws_mat)
nis_weights = ws_mat / T.sum(ws_mat, axis=1, keepdims=True)
nis_weights = theano.gradient.disconnected_grad(nis_weights)

iwae_obs_costs = -1.0 * (T.sum((nis_weights * log_ws_mat), axis=1) - \
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
vae_obs_klds = sum([T.mean(mod_kld.reshape((batch_size, iwae_samples)), axis=1) \
                     for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

vae_bound = vae_nll_cost + vae_kld_cost

######################################################
# Get functions for free sampling and reconstruction #
######################################################
# get simple reconstruction, for other purposes
im_rd = inf_gen_model.apply_im(Xg)
Xg_recon = im_rd['td_output']
# run an un-grounded pass through generative stuff for sampling from model
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
Xd_model = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)


# build training cost and update functions
t = time()
print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg], Xg_recon)
sample_func = theano.function([Z0], Xd_model)
test_recons = recon_func(train_transform(Xtr[0:100,:])) # cheeky model implementation test
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
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/EVAL.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

Xva_blocks = [Xva] #np.split(Xva, 2, axis=0)
for epoch in range(2):
    epoch_vae_cost = 0.0
    epoch_iwae_cost = 0.0
    for block_num, Xva_block in enumerate(Xva_blocks):
        Xva_block = shuffle(Xva_block)
        obs_count = Xva_block.shape[0]
        g_epoch_costs = [0. for c in g_basic_costs]
        g_batch_count = 0.
        for imb in tqdm(iter_data(Xva_block, size=nbatch), total=obs_count/nbatch):
            # transform validation batch to "image format"
            imb_img = floatX( train_transform(imb) )
            # evaluate costs
            g_result = g_eval_func(imb_img)
            # evaluate costs more thoroughly
            iwae_bounds = iwae_multi_eval(imb_img, 50*25,
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
        g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                     for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
        str2 = "    {}".format(" ".join(g_bc_strs))
        joint_str = "\n".join([str1, str2])
        print(joint_str)
        out_file.write(joint_str+"\n")
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
    print("EPOCH {0:d} -- vae: {1:.2f}, iwae: {2:.2f}".format(epoch, epoch_vae_cost, epoch_iwae_cost))







##############
# EYE BUFFER #
##############
