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
desc = 'test_fc_dyn_bin_old_sc_deeper_bypass_long_train'
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
nbatch = 100      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 64          # # of dim for Z0
nz1 = 64          # # of dim for Z1
ngf = 64          # base # of filters for conv layers in generative stuff
ngfc = 64         # number of filters in top-most fc layer
scf = 256
nx = npx*npx*nc   # # of dimensions in X
niter = 500       # # of iter at starting learning rate
niter_decay = 3000 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_fc = True     # whether to use "internal" conv layers in gen/disc networks
use_bn = True     # whether to use batch normalization throughout the model
act_func = 'lrelu' # activation func to use where they can be selected
iwae_samples = 1  # number of samples to use in MEN bound
noise_std = 0.1   # amount of noise to inject in BU and IM modules
use_td_noise = True # whether to use noise in TD pass
use_bu_noise = True # whether to use noise in BU pass

ntrain = Xtr.shape[0]


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
Xg = T.matrix()  # symbolic var for inputs to bottom-up inference network
Z0 = T.matrix()   # symbolic var for "noise" inputs to the generative stuff

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
# parameter regularization part of cost
vae_reg_cost = 2e-5 * sum([T.sum(p**2.0) for p in g_params])
# run an inference and reconstruction pass through the generative stuff
im_res_dict = inf_gen_model.apply_im(Xg, noise=noise)
Xg_recon = im_res_dict['td_output']
kld_dict = im_res_dict['kld_dict']
log_p_z = sum(im_res_dict['log_p_z'])
log_q_z = sum(im_res_dict['log_q_z'])

#log_p_x = T.sum(bce(T.flatten(Xg_recon, 2), T.flatten(Xg, 2)), axis=1)

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

# get simple reconstruction, for other purposes
im_rd = inf_gen_model.apply_im(Xg)
Xgr2 = im_rd['td_output']

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
recon_func = theano.function([Xg], Xgr2)
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
g_train_func = theano.function([Xg], g_cost_outputs, updates=g_updates)   # train inference and generator
i_train_func = theano.function([Xg], g_cost_outputs, updates=inf_updates) # train inference only
g_eval_func = theano.function([Xg], g_cost_outputs)                       # evaluate model costs
print "{0:.2f} seconds to compile theano functions".format(time()-t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

print("EXPERIMENT: {}".format(desc.upper()))

n_check = 0
n_updates = 0
t = time()
lam_vae.set_value(floatX([0.5]))
kld_weights = np.linspace(0.0,1.0,25)
sample_z0mb = rand_gen(size=(200, nz0)) # root noise for visualizing samples
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    Xva = shuffle(Xva)
    # mess with the KLd cost
    if ((epoch-1) < len(kld_weights)):
        lam_kld.set_value(floatX([kld_weights[epoch-1]]))
    #lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    i_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    gen_grad_norms = []
    inf_grad_norms = []
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    i_batch_count = 0.
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
        g_result = g_train_func(floatX(imb_img))
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1.*g_result[3])
        vae_klds.append(1.*g_result[4])
        gen_grad_norms.append(1.*g_result[5])
        inf_grad_norms.append(1.*g_result[6])
        batch_obs_costs = g_result[7]
        batch_layer_klds = g_result[8]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # train inference model on samples from the generator
        #if epoch > 25:
        #    smb_img = binarize_data(sample_func(rand_gen(size=(100, nz0))))
        #    i_result = i_train_func(smb_img)
        #    i_epoch_costs = [(v1 + v2) for v1, v2 in zip(i_result[:5], i_epoch_costs)]
        i_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            noise.set_value(floatX([0.0]))
            v_result = g_eval_func(vmb_img)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:6], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 10) or (epoch == 50) or (epoch == 100) or (epoch == 250):
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
    i_epoch_costs = [(c / i_batch_count) for c in i_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    i_bc_strs = ["{0:s}: {1:.2f},".format(c_name, i_epoch_costs[c_idx]) \
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2i = " ".join(i_bc_strs)
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
    joint_str = "\n".join([str1, str2, str2i, str3, str4, str5, str6, str7, str8])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()





##############
# EYE BUFFER #
##############
