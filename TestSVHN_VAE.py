import os
import json
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.costs import log_prob_gaussian
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import load_svhn

#
# Phil's business
#
from MatryoshkaModules import BasicConvModule, GenConvModule, \
                              GenConvDblResModule, GenConvResModule, \
                              GenFCModule, InfConvMergeModule, \
                              InfFCModule, BasicConvResModule
from MatryoshkaNetworks import InfGenModel

# path for dumping experiment info and fetching dataset
EXP_DIR = "./svhn"
DATA_SIZE = 250000

# setup paths for dumping diagnostic info
desc = 'test_resnet_vae'
model_dir = "{}/models/{}".format(EXP_DIR, desc)
sample_dir = "{}/samples/{}".format(EXP_DIR, desc)
log_dir = "{}/logs".format(EXP_DIR)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# locations of 32x32 SVHN dataset
tr_file = "{}/data/svhn_train.pkl".format(EXP_DIR)
te_file = "{}/data/svhn_test.pkl".format(EXP_DIR)
ex_file = "{}/data/svhn_extra.pkl".format(EXP_DIR)
# load dataset (load more when using adequate computers...)
data_dict = load_svhn(tr_file, te_file, ex_file=ex_file, ex_count=DATA_SIZE)

# stack data into a single array and rescale it into [-1,1]
Xtr = np.concatenate([data_dict['Xtr'], data_dict['Xte'], data_dict['Xex']], axis=0)
del data_dict
Xtr = Xtr - np.min(Xtr)
Xtr = Xtr / np.max(Xtr)
Xtr = 2.0 * (Xtr - 0.5)
Xtr_std = np.std(Xtr, axis=0, keepdims=True)
Xtr_var = Xtr_std**2.0

set_seed(1)       # seed for shared rngs
l2 = 1.0e-5       # l2 weight decay
b1 = 0.9          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz0 = 64          # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngfc = 256        # # of filters in fully connected layers
ngf = 64          # # of filters in first convolutional layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
all_rand = True   # whether to use stochastic variables at multiple scales
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_annealing = True # whether to use "annealing" of the target distribution

ntrain = Xtr.shape[0]


def train_transform(X):
    # transform vectorized observations into convnet inputs
    return X.reshape(-1, nc, npx, npx).transpose(0, 1, 2, 3)

def draw_transform(X):
    # transform vectorized observations into drawable images
    X = (X + 1.0) * 127.0
    return X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)

def rand_gen(size, noise_type='normal'):
    if noise_type == 'normal':
        r_vals = floatX(np_rng.normal(size=size))
    elif noise_type == 'uniform':
        r_vals = floatX(np_rng.uniform(size=size, low=-1.0, high=1.0))
    else:
        assert False, "unrecognized noise type!"
    return r_vals

def gauss_blur(x, x_std, w_x, w_g):
    """
    Add gaussian noise to x, with rescaling to keep variance constant w.r.t.
    the initial variance of x (in x_var). w_x and w_g should be weights for
    a convex combination.
    """
    g_std = np.sqrt( (x_std * (1. - w_x)**2.) / (w_g**2. + 1e-4) )
    g_noise = g_std * np_rng.normal(size=x.shape)
    x_blurred = w_x*x + w_g*g_noise
    return floatX(x_blurred)

# draw some examples from training set
color_grid_vis(draw_transform(Xtr[0:200]), (10, 20), "{}/Xtr.png".format(sample_dir))

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


#########################################
# Setup the top-down processing modules #
# -- these do generation                #
#########################################

td_module_1 = \
GenFCModule(
    rand_dim=nz0,
    out_shape=(ngf*4, 2, 2),
    fc_dim=ngfc,
    num_layers=2,
    apply_bn_1=True,
    apply_bn_2=True,
    mod_name='td_mod_1'
) # output is (batch, ngf*4, 2, 2)

td_module_2 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    rand_chans=nz1,
    use_rand=False,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_2'
) # output is (batch, ngf*4, 4, 4)

td_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=ngf,
    rand_chans=nz1,
    use_rand=all_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_3'
) # output is (batch, ngf*2, 8, 8)

td_module_4 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=ngf,
    rand_chans=nz1,
    use_rand=False,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_4'
) # output is (batch, ngf*2, 16, 16)

td_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=ngf,
    rand_chans=nz1,
    use_rand=all_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='td_mod_5'
) # output is (batch, ngf*1, 32, 32)

td_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*1),
    out_chans=nc,
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='td_mod_6'
) # output is (batch, c, 32, 32)

# modules must be listed in "evaluation order"
td_modules = [td_module_1, td_module_2, td_module_3,
              td_module_4, td_module_5, td_module_6]

##########################################
# Setup the bottom-up processing modules #
# -- these do inference                  #
##########################################

bu_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=(ngf*1),
    apply_bn=True,
    stride='single',
    act_func='relu',
    mod_name='bu_mod_6'
) # output is (batch, ngf*1, 32, 32)

bu_module_5 = \
BasicConvResModule(
    in_chans=(ngf*1),
    out_chans=(ngf*2),
    conv_chans=ngf,
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_5'
) # output is (batch, ngf*2, 16, 16)

bu_module_4 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*2),
    conv_chans=ngf,
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_4'
) # output is (batch, ngf*2, 8, 8)

bu_module_3 = \
BasicConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*4),
    conv_chans=ngf,
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_3'
) # output is (batch, ngf*4, 4, 4)

bu_module_2 = \
BasicConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*4),
    conv_chans=(ngf*2),
    use_conv=use_conv,
    stride='double',
    act_func='relu',
    mod_name='bu_mod_2'
) # output is (batch, ngf*4, 2, 2)

bu_module_1 = \
InfFCModule(
    bu_chans=(ngf*4*2*2),
    fc_chans=ngfc,
    rand_chans=nz0,
    use_fc=True,
    mod_name='bu_mod_1'
) # output is (batch, nz0), (batch, nz0)

# modules must be listed in "evaluation order"
bu_modules = [bu_module_6, bu_module_5, bu_module_4,
              bu_module_3, bu_module_2, bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_3 = \
InfConvMergeModule(
    td_chans=(ngf*4),
    bu_chans=(ngf*4),
    rand_chans=nz1,
    conv_chans=(ngf*2),
    use_conv=use_conv,
    mod_name='im_mod_3'
) # merge input to td_mod_3 and output of bu_mod_3, to place a distribution
  # over the rand_vals used in td_mod_3.

im_module_5 = \
InfConvMergeModule(
    td_chans=(ngf*2),
    bu_chans=(ngf*2),
    rand_chans=nz1,
    conv_chans=(ngf*1),
    use_conv=use_conv,
    mod_name='im_mod_5'
) # merge input to td_mod_5 and output of bu_mod_5, to place a distribution
  # over the rand_vals used in td_mod_5.

im_modules = [im_module_3, im_module_5]

#
# Setup a description for where to get conditional distributions from. When
# there's no info here for a particular top-down module, we won't pass any
# random variables explicitly into the module, which will cause the module to
# generate its own random variables (unconditionally). When a "bu_module" is
# provided and an "im_module" is not, the conditional distribution is specified
# directly by the bu_module's output, and no merging (via an im_module) is
# required.
#
merge_info = {
    'td_mod_1': {'bu_module': 'bu_mod_1', 'im_module': None},
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'},
}

# construct the "wrapper" object for managing all our modules
inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=tanh
)


####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(np.ones((1,)).astype(theano.config.floatX))
obs_logvar = sharedX(np.zeros((1,)).astype(theano.config.floatX))
bounded_logvar = 6.0 * tanh((1.0/6.0) * obs_logvar)
model_params = [obs_logvar] + inf_gen_model.params

X = T.tensor4()   # symbolic var for real inputs to mega deep, convolutional generatotron
Z0 = T.matrix()

# draw sample reconstructons from the generatotron, and compute some KLds too.
td_output, kld_dicts = inf_gen_model.apply_im(X)

nll_costs = -1.0 * log_prob_gaussian(T.flatten(X, 2), T.flatten(td_output, 2),
                                     log_vars=bounded_logvar[0])
layer_klds = [T.sum(kld_i, axis=1) for kld_i in kld_dicts.values()]
kld_costs = sum(layer_klds)

nll_cost = T.mean(nll_costs)
kld_cost = lam_kld[0] * T.mean(kld_costs)
reg_cost = 1e-6 * sum([T.sum(p**2.0) for p in model_params])
total_cost = nll_cost + kld_cost + reg_cost

# compile a theano function strictly for sampling reconstructions from generatotron
trial_func = theano.function([X], [td_output, total_cost])
# TEMP TEST FOR MODEL ARCHITECTURE
x_batch = train_transform(Xtr[0:100,:])
batch_output = trial_func(x_batch)
print("TEST -- total_cost: {0:.4f}".format(1.0*batch_output[-1]))

# draw samples from the generator, with initial random vals provided by the user
td_inputs = [Z0] + [None for td_mod in td_modules[1:]]
XIZ0 = inf_gen_model.apply_td(rand_vals=td_inputs, batch_size=None)

# stuff for performing updates
lrt = sharedX(lr)
p_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)

t = time()
print("Computing gradients...")
model_updates = p_updater(model_params, total_cost)
print("Compiling sampling function...")
sample_func = theano.function([Z0], XIZ0)
print("Compiling training function...")
train_func = theano.function([X], [total_cost, nll_cost, kld_cost, reg_cost],
                             updates=model_updates)
print "{0:.2f} seconds to compile theano functions".format(time()-t)


log_name = "{}/RESULTS.txt".format(sample_dir)
out_file = open(log_name, 'wb')

n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
sample_z0mb = rand_gen(size=(200, nz0)) # noise samples for top generator module
for epoch in range(1, niter+niter_decay+1):
    Xtr = shuffle(Xtr)
    scale = min(1.0, (epoch/10.0))
    lam_kld.set_value(np.asarray([scale]).astype(theano.config.floatX))
    total_cost = 0.
    nll_cost = 0.
    kld_cost = 0.
    reg_cost = 0.
    batch_count = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        imb = train_transform(imb)
        # compute model cost and apply update
        result = train_func(imb)
        total_cost += result[0]
        nll_cost += result[1]
        kld_cost += result[2]
        reg_cost += result[3]
        batch_count += 1
        n_updates += 1
        n_examples += len(imb)
    str1 = "Epoch {}:".format(epoch)
    str2 = "    total_cost: {0:.4f}, nll_cost: {1:.4f}, kld_cost: {2:.4f}, reg_cost: {3:.4f}".format( \
            (total_cost/batch_count), (nll_cost/batch_count), (kld_cost/batch_count), (reg_cost/batch_count))
    joint_str = "\n".join([str1, str2])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    # generate some samples from the model, for visualization
    samples = np.asarray(sample_func(sample_z0mb))
    color_grid_vis(draw_transform(samples), (10, 20), "{}/{}.png".format(sample_dir, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
        joblib.dump([p.get_value() for p in gen_params], "{}/{}_gen_params.jl".format(model_dir, n_epochs))
        joblib.dump([p.get_value() for p in disc_params], "{}/{}_disc_params.jl".format(model_dir, n_epochs))
