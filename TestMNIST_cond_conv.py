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
from MatryoshkaModules import \
    BasicConvModule, GenTopModule, InfTopModule, \
    GenConvPertModule, BasicConvPertModule, \
    GenConvGRUModule, InfConvMergeModuleIMS
from MatryoshkaNetworks import CondInfGenModel

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

# setup paths for dumping diagnostic info
desc = 'test_cond_impute_2x_upsampling'
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
nz0 = 32           # # of dim for Z0
nz1 = 4            # # of dim for Z1
ngf = 32           # base # of filters for conv layers in generative stuff
ngfc = 128         # # of filters in fully connected layers of generative stuff
nx = npx * npx * nc  # # of dimensions in X
niter = 150        # # of iter at starting learning rate
niter_decay = 250  # # of iter to linearly decay learning rate to zero
multi_rand = True  # whether to use stochastic variables at multiple scales
use_conv = True    # whether to use "internal" conv layers in gen/disc networks
use_bn = True      # whether to use batch normalization throughout the model
act_func = 'lrelu'  # activation func to use where they can be selected
noise_std = 0.0     # amount of noise to inject in BU and IM modules
use_bu_noise = False
use_td_noise = False
inf_mt = 0
use_td_cond = False
depth_7x7 = 2
depth_14x14 = 2

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

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

# FC -> (7, 7)
td_module_1 = \
    GenTopModule(
        rand_dim=nz0,
        out_shape=(ngf * 2, 7, 7),
        fc_dim=ngfc,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='td_mod_1')

# grow the (7, 7) -> (7, 7) part of network
td_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'td_mod_2{}'.format(alphabet[i])
    new_module = \
        GenConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            rand_chans=nz1,
            filt_shape=(3, 3),
            use_rand=multi_rand,
            use_conv=use_conv,
            apply_bn=use_bn,
            act_func=act_func,
            us_stride=1,
            mod_name=mod_name)
    td_modules_7x7.append(new_module)
# manual stuff for parameter sharing....

# (7, 7) -> (14, 14)
td_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
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
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            rand_chans=nz1,
            filt_shape=(3, 3),
            use_rand=multi_rand,
            use_conv=use_conv,
            apply_bn=use_bn,
            act_func=act_func,
            us_stride=1,
            mod_name=mod_name)
    td_modules_14x14.append(new_module)
# manual stuff for parameter sharing....

# (14, 14) -> (28, 28)
td_module_5 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 2),
        out_chans=(ngf * 1),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='td_mod_5')

# (28, 28) -> (28, 28)
td_module_6 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 1),
        out_chans=nc,
        apply_bn=False,
        use_noise=False,
        stride='single',
        act_func='ident',
        mod_name='td_mod_6')

# modules must be listed in "evaluation order"
td_modules = [td_module_1] + \
             td_modules_7x7 + \
             [td_module_3] + \
             td_modules_14x14 + \
             [td_module_5, td_module_6]

##########################################
# Setup the bottom-up processing modules #
# -- these do generation inference       #
##########################################

# (7, 7) -> FC
bu_module_1 = \
    InfTopModule(
        bu_chans=(ngf * 2 * 7 * 7),
        fc_chans=ngfc,
        rand_chans=nz0,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='bu_mod_1')

# grow the (7, 7) -> (7, 7) part of network
bu_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'bu_mod_2{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_7x7.append(new_module)
bu_modules_7x7.reverse()  # reverse, to match "evaluation order"

# (14, 14) -> (7, 7)
bu_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_3')

# grow the (14, 14) -> (14, 14) part of network
bu_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'bu_mod_4{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_14x14.append(new_module)
bu_modules_14x14.reverse()  # reverse, to match "evaluation order"

# (28, 28) -> (14, 14)
bu_module_5 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 1),
        out_chans=(ngf * 2),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_5')

# (28, 28) -> (28, 28)
bu_module_6 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(nc + 1),
        out_chans=(ngf * 1),
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name='bu_mod_6')

# modules must be listed in "evaluation order"
bu_modules_gen = [bu_module_6, bu_module_5] + \
                 bu_modules_14x14 + \
                 [bu_module_3] + \
                 bu_modules_7x7 + \
                 [bu_module_1]


##########################################
# Setup the bottom-up processing modules #
# -- these do inference inference        #
##########################################

# (7, 7) -> FC
bu_module_1 = \
    InfTopModule(
        bu_chans=(ngf * 2 * 7 * 7),
        fc_chans=ngfc,
        rand_chans=nz0,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='bu_mod_1')

# grow the (7, 7) -> (7, 7) part of network
bu_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'bu_mod_2{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_7x7.append(new_module)
bu_modules_7x7.reverse()  # reverse, to match "evaluation order"

# (14, 14) -> (7, 7)
bu_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_3')

# grow the (14, 14) -> (14, 14) part of network
bu_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'bu_mod_4{}'.format(alphabet[i])
    new_module = \
        BasicConvPertModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            conv_chans=(ngf * 2),
            filt_shape=(3, 3),
            use_conv=use_conv,
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name=mod_name)
    bu_modules_14x14.append(new_module)
bu_modules_14x14.reverse()  # reverse, to match "evaluation order"

# (28, 28) -> (14, 14)
bu_module_5 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(ngf * 1),
        out_chans=(ngf * 2),
        apply_bn=use_bn,
        stride='double',
        act_func=act_func,
        mod_name='bu_mod_5')

# (28, 28) -> (28, 28)
bu_module_6 = \
    BasicConvModule(
        filt_shape=(3, 3),
        in_chans=(2 * (nc + 1)),
        out_chans=(ngf * 1),
        apply_bn=use_bn,
        stride='single',
        act_func=act_func,
        mod_name='bu_mod_6')

# modules must be listed in "evaluation order"
bu_modules_inf = [bu_module_6, bu_module_5] + \
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
        out_shape=(ngf * 2, 7, 7),
        fc_dim=ngfc,
        use_fc=True,
        use_sc=False,
        apply_bn=use_bn,
        act_func=act_func,
        mod_name='im_mod_1')

# grow the (7, 7) -> (7, 7) part of network
im_modules_7x7 = []
for i in range(depth_7x7):
    mod_name = 'im_mod_2{}'.format(alphabet[i])
    new_module = \
        InfConvMergeModuleIMS(
            td_chans=(ngf * 2),
            bu_chans=(ngf * 2),
            im_chans=(ngf * 2),
            rand_chans=nz1,
            conv_chans=(ngf * 2),
            use_conv=True,
            use_td_cond=use_td_cond,
            apply_bn=use_bn,
            mod_type=inf_mt,
            act_func=act_func,
            mod_name=mod_name)
    im_modules_7x7.append(new_module)

# (7, 7) -> (14, 14)
im_module_3 = \
    BasicConvModule(
        in_chans=(ngf * 2),
        out_chans=(ngf * 2),
        filt_shape=(3, 3),
        apply_bn=use_bn,
        stride='half',
        act_func=act_func,
        mod_name='im_mod_3')

# grow the (14, 14) -> (14, 14) part of network
im_modules_14x14 = []
for i in range(depth_14x14):
    mod_name = 'im_mod_4{}'.format(alphabet[i])
    new_module = \
        InfConvMergeModuleIMS(
            td_chans=(ngf * 2),
            bu_chans=(ngf * 2),
            im_chans=(ngf * 2),
            rand_chans=nz1,
            conv_chans=(ngf * 2),
            use_conv=True,
            use_td_cond=use_td_cond,
            apply_bn=use_bn,
            mod_type=inf_mt,
            act_func=act_func,
            mod_name=mod_name)
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
        im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
    if i < (depth_7x7 - 1):
        bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
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
        im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
    if i < (depth_14x14 - 1):
        bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
    # add entry for this TD module
    merge_info[td_mod_name] = {
        'td_type': td_type, 'im_module': im_mod_name,
        'bu_source': bu_src_name, 'im_source': im_src_name
    }


# transforms to apply to generator outputs
def output_transform(x):
    output = sigmoid(T.clip(x, -15.0, 15.0))
    return output


def output_noop(x):
    output = x
    return output


# construct the "wrapper" object for managing all our modules
inf_gen_model = CondInfGenModel(
    td_modules=td_modules,
    bu_modules_gen=bu_modules_gen,
    im_modules_gen=im_modules,
    bu_modules_inf=bu_modules_inf,
    im_modules_inf=im_modules,
    merge_info=merge_info,
    output_transform=output_noop)

# inf_gen_model.load_params(inf_gen_param_file)

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
X_init = sharedX(floatX(np.zeros((1, nc, npx, npx))))  # default "initial state"
noise = sharedX(floatX([noise_std]))
gen_params = inf_gen_model.gen_params
inf_params = inf_gen_model.inf_params
all_params = inf_gen_model.all_params + [X_init]

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg_gen = T.tensor4()  # input to generator, with some parts masked out
Xm_gen = T.tensor4()  # mask indicating parts that are masked out
Xg_inf = T.tensor4()  # complete observation, for input to inference net
Xm_inf = T.tensor4()  # mask for which bits to predict
# get the full inputs to the generator and inferencer networks
Xa_gen = T.concatenate([Xg_gen, Xm_gen], axis=1)
Xa_inf = T.concatenate([Xg_gen, Xm_gen, Xg_inf, Xm_inf], axis=1)

##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################

# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in all_params])

# feed all masked inputs through the inference network
im_res_dict_inf = inf_gen_model.apply_im(Xa_inf, mode='inf')
z_inf = im_res_dict_inf['z_dict']
logz_inf = im_res_dict_inf['logz_dict']
Xg_recon = sigmoid(T.clip(im_res_dict_inf['td_output'], -15., 15.))

# evaluate log-likelihood of latent samples under the conditonal prior
im_res_dict_gen = inf_gen_model.apply_im(Xa_gen, mode='gen', z_vals=z_inf)
logz_gen = im_res_dict_gen['logz_dict']
# sample freely from conditional prior
im_res_dict_gen = inf_gen_model.apply_im(Xa_gen, mode='gen')
Xg_sampl = sigmoid(T.clip(im_res_dict_gen['td_output'], -15., 15.))

# compute masked reconstruction error from final step.
log_p_x = T.sum(log_prob_bernoulli(
                T.flatten(Xg_inf, 2), T.flatten(Xg_recon, 2),
                mask=T.flatten((1. - Xm_gen), 2), do_sum=False),
                axis=1)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# convert KL dict to aggregate KLds over inference steps
kl_by_td_mod = {td_mod_name: (logz_inf[td_mod_name] - logz_gen[td_mod_name]) for
                td_mod_name in logz_inf.keys()}
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

#
# test the model implementation
#
inputs = [Xg_gen, Xm_gen, Xg_inf, Xm_inf]
outputs = [full_cost]
print('Compiling test function...')
test_func = theano.function(inputs, outputs)


# grab data to feed into the model
def make_model_input(x_in):
    xg_gen, xg_inf, xm_gen = \
        get_downsampling_masks(x_in, im_shape=(28, 28),
                               im_chans=1, data_mean=Xmu)
    # construct_masked_data(x_in, drop_prob=0.5, occ_dim=None,
    #                       occ_count=1, data_mean=Xmu)
    xm_inf = 1. - xm_gen
    xg_gen = train_transform(xg_gen)
    xm_gen = train_transform(xm_gen)
    xg_inf = train_transform(xg_inf)
    xm_inf = train_transform(xm_inf)
    return xg_gen, xm_gen, xg_inf, xm_inf

# test the model implementation
model_input = make_model_input(Xtr[0:100, :])
test_out = test_func(*model_input)
print('DONE.')

#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

# stuff for performing updates
lrt = sharedX(0.001)
b1t = sharedX(0.9)
updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-4, clipnorm=1000.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
all_updates, all_grads = updater(all_params, full_cost, return_grads=True)

print("Compiling sampling and reconstruction functions...")
recon_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], Xg_recon)
sampl_func = theano.function([Xg_gen, Xm_gen], Xg_sampl)
model_input = make_model_input(Xtr[0:100, :])
test_recons = recon_func(*model_input)  # cheeky model implementation test
test_sampls = sampl_func(model_input[0], model_input[1])

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
                               updates=all_updates)
g_eval_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], g_cost_outputs)
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
        noise.set_value(floatX([noise_std]))
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
            noise.set_value(floatX([0.0]))
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
