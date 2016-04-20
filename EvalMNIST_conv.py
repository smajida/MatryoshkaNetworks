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
from MatryoshkaModules import BasicConvModule, GenTopModule, InfTopModule, \
                              GenConvPertModule, BasicConvPertModule, \
                              InfConvMergeModuleIMS
from MatryoshkaNetworks import InfGenModel

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_conv_new_matnet_ims_im_res_late_cond_5deep_3'
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
else:
    dataset = load_udm("{}mnist.pkl.gz".format(data_path), to_01=True)
    Xtr = dataset[0][0]
    Xva = dataset[1][0]
    Xte = dataset[2][0]
    Xtr = np.concatenate([Xtr, Xva], axis=0).copy()
    Xva = Xte

set_seed(123)     # seed for shared rngs
nc = 1            # # of channels in image
nbatch = 10       # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 32          # # of dim for Z0
nz1 = 4           # # of dim for Z1
ngf = 32          # base # of filters for conv layers in generative stuff
ngfc = 128        # # of filters in fully connected layers of generative stuff
nx = npx*npx*nc   # # of dimensions in X
niter = 150       # # of iter at starting learning rate
niter_decay = 150 # # of iter to linearly decay learning rate to zero
multi_rand = True # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_bn = False     # whether to use batch normalization throughout the model
act_func = 'lrelu' # activation func to use where they can be selected
noise_std = 0.0    # amount of noise to inject in BU and IM modules
iwae_samples = 10
use_bu_noise = False
use_td_noise = False
inf_mt = 1
use_td_cond = False
depth_7x7 = 5
depth_14x14 = 5

fine_tune_inf_net = True

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

ntrain = Xva.shape[0]


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

# FC -> (7, 7)
td_module_1 = \
GenTopModule(
    rand_dim=nz0,
    out_shape=(ngf*2, 7, 7),
    fc_dim=ngfc,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='td_mod_1'
)

# grow the (7, 7) -> (7, 7) part of network
td_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'td_mod_2{}'.format(alphabet[i])
    new_module = \
    GenConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        rand_chans=nz1,
        filt_shape=(3,3),
        use_rand=multi_rand,
        use_conv=use_conv,
        apply_bn=use_bn,
        act_func=act_func,
        us_stride=1,
        mod_name=mod_name
    )
    td_modules_7x7.append(new_module)
# manual stuff for parameter sharing....

# (7, 7) -> (14, 14)
td_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='td_mod_3'
)

# grow the (14, 14) -> (14, 14) part of network
td_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'td_mod_4{}'.format(alphabet[i])
    new_module = \
    GenConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        rand_chans=nz1,
        filt_shape=(3,3),
        use_rand=multi_rand,
        use_conv=use_conv,
        apply_bn=use_bn,
        act_func=act_func,
        us_stride=1,
        mod_name=mod_name
    )
    td_modules_14x14.append(new_module)
# manual stuff for parameter sharing....

# (14, 14) -> (28, 28)
td_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='td_mod_5'
)

# (28, 28) -> (28, 28)
td_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
    apply_bn=False,
    use_noise=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_6'
)

# modules must be listed in "evaluation order"
td_modules = [td_module_1] + \
             td_modules_7x7 + \
             [td_module_3] + \
             td_modules_14x14 + \
             [td_module_5, td_module_6]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

# (7, 7) -> FC
bu_module_1 = \
InfTopModule(
    bu_chans=(ngf*2*7*7),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='bu_mod_1'
)

# grow the (7, 7) -> (7, 7) part of network
bu_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'bu_mod_2{}'.format(alphabet[i])
    new_module = \
    BasicConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        filt_shape=(3,3),
        use_conv=use_conv,
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name=mod_name
    )
    bu_modules_7x7.append(new_module)
bu_modules_7x7.reverse() # reverse, to match "evaluation order"

# (14, 14) -> (7, 7)
bu_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_3'
)

# grow the (14, 14) -> (14, 14) part of network
bu_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'bu_mod_4{}'.format(alphabet[i])
    new_module = \
    BasicConvPertModule(
        in_chans=(ngf*2),
        out_chans=(ngf*2),
        conv_chans=(ngf*2),
        filt_shape=(3,3),
        use_conv=use_conv,
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name=mod_name
    )
    bu_modules_14x14.append(new_module)
bu_modules_14x14.reverse() # reverse, to match "evaluation order"

# (28, 28) -> (14, 14)
bu_module_5 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    apply_bn=use_bn,
    stride='double',
    act_func=act_func,
    mod_name='bu_mod_5'
)

# (28, 28) -> (28, 28)
bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=use_bn,
    stride='single',
    act_func=act_func,
    mod_name='bu_mod_6'
)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_6, bu_module_5] + \
             bu_modules_14x14 + \
             [bu_module_3] + \
             bu_modules_7x7 + \
             [bu_module_1]


#########################################
# Setup the information merging modules #
#########################################

# FC -> (7, 7)
im_module_1 = \
GenTopModule(
    rand_dim=nz0,
    out_shape=(ngf*2, 7, 7),
    fc_dim=ngfc,
    use_fc=True,
    use_sc=False,
    apply_bn=use_bn,
    act_func=act_func,
    mod_name='im_mod_1'
)

# grow the (7, 7) -> (7, 7) part of network
im_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'im_mod_2{}'.format(alphabet[i])
    new_module = \
    InfConvMergeModuleIMS(
        td_chans=(ngf*2),
        bu_chans=(ngf*2),
        im_chans=(ngf*2),
        rand_chans=nz1,
        conv_chans=(ngf*2),
        use_conv=True,
        use_td_cond=use_td_cond,
        apply_bn=use_bn,
        mod_type=inf_mt,
        act_func=act_func,
        mod_name=mod_name
    )
    im_modules_7x7.append(new_module)

# (7, 7) -> (14, 14)
im_module_3 = \
BasicConvModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    filt_shape=(3,3),
    apply_bn=use_bn,
    stride='half',
    act_func=act_func,
    mod_name='im_mod_3'
)

# grow the (14, 14) -> (14, 14) part of network
im_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'im_mod_4{}'.format(alphabet[i])
    new_module = \
    InfConvMergeModuleIMS(
        td_chans=(ngf*2),
        bu_chans=(ngf*2),
        im_chans=(ngf*2),
        rand_chans=nz1,
        conv_chans=(ngf*2),
        use_conv=True,
        use_td_cond=use_td_cond,
        apply_bn=use_bn,
        mod_type=inf_mt,
        act_func=act_func,
        mod_name=mod_name
    )
    im_modules_14x14.append(new_module)

im_modules = [im_module_1] + \
             im_modules_7x7 + \
             [im_module_3] + \
             im_modules_14x14

#
# Setup a description for where to get conditional distributions from.
#
merge_info = {
    'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                 'bu_source': 'bu_mod_1', 'im_source': None},

    'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                 'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

    'td_mod_5': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None},
    'td_mod_6': {'td_type': 'pass', 'im_module': None,
                 'bu_source': None, 'im_source': None}
}

# add merge_info entries for the modules with latent variables
for i in range(depth_7x7):
    td_type = 'cond'
    td_mod_name = 'td_mod_2{}'.format(alphabet[i])
    im_mod_name = 'im_mod_2{}'.format(alphabet[i])
    im_src_name = 'im_mod_1'
    bu_src_name = 'bu_mod_3'
    if i > 0:
        im_src_name = 'im_mod_2{}'.format(alphabet[i-1])
    if i < (depth_7x7 - 1):
        bu_src_name = 'bu_mod_2{}'.format(alphabet[i+1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }
for i in range(depth_14x14):
    td_type = 'cond'
    td_mod_name = 'td_mod_4{}'.format(alphabet[i])
    im_mod_name = 'im_mod_4{}'.format(alphabet[i])
    im_src_name = 'im_mod_3'
    bu_src_name = 'bu_mod_5'
    if i > 0:
        im_src_name = 'im_mod_4{}'.format(alphabet[i-1])
    if i < (depth_14x14 - 1):
        bu_src_name = 'bu_mod_4{}'.format(alphabet[i+1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }


# construct the "wrapper" object for managing all our modules
output_transform = lambda x: sigmoid(T.clip(x, -15.0, 15.0))
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    sc_modules=[],
    merge_info=merge_info,
    output_transform=output_transform,
    use_sc=False
)

###################
# LOAD PARAMETERS #
###################
inf_gen_model.load_params(inf_gen_param_file)

#################################################
#################################################
##                                             ##
## FINE TUNE THE INFERENCE NETWORK ON TEST SET ##
##                                             ##
#################################################
#################################################
if fine_tune_inf_net:
    ####################################
    # Setup the optimization objective #
    ####################################
    lam_kld = sharedX(floatX([1.0]))
    noise = sharedX(floatX([noise_std]))
    inf_params = inf_gen_model.inf_params

    ##########################################################
    # CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
    ##########################################################
    Xg = T.tensor4()  # symbolic var for inputs to bottom-up inference network
    # parameter regularization part of cost
    vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in inf_params])

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
    print "{0:.2f} seconds to compile theano functions".format(time()-t)

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
        for imb in tqdm(iter_data(Xva, size=100), total=ntrain/100):
            # transform training batch to "image format"
            imb_img = train_transform(imb)
            # train vae on training batch
            noise.set_value(floatX([noise_std]))
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
        g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx]) \
                     for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
        str2 = " ".join(g_bc_strs)
        joint_str = "\n".join([str1, str2])
        print(joint_str)
        out_file.write(joint_str+"\n")
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

Xva_blocks = np.split(Xva, 5, axis=0)
for epoch in range(5):
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
            iwae_bounds = iwae_multi_eval(imb_img, 500,
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
    str1 = "EPOCH {0:d} -- vae: {1:.2f}, iwae: {2:.2f}".format(epoch, epoch_vae_cost, epoch_iwae_cost)
    print(str1)
    out_file.write(str1+"\n")
    out_file.flush()






##############
# EYE BUFFER #
##############
