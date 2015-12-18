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

bu_modules = [bu_module_1, bu_module_2, bu_module_3,
              bu_module_4, bu_module_5, bu_module_6]

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

# setup a description for where to get conditional distributions from
merge_info = {
    'td_mod_1': {'bu_module': 'bu_mod_1', 'im_module': None},
    'td_mod_5': {'bu_module': 'bu_mod_5', 'im_module': 'im_mod_5'},
    'td_mod_3': {'bu_module': 'bu_mod_3', 'im_module': 'im_mod_3'},
}

# construct the "wrapper" object for managing all our modules

inf_gen_model = InfGenModel(
    bu_modules=bu_modules,
    td_modules=td_modules,
    im_modules=im_modules,
    merge_info=merge_info,
    output_transform=tanh
)

#
# TODO: finish testing code.
#

#
# ####################################
# # Setup the optimization objective #
# ####################################
#
# X = T.tensor4()   # symbolic var for real inputs to discriminator
# Z0 = T.matrix()   # symbolic var for rand values to pass into generator
# Xer = T.tensor4() # symbolic var for samples from experience replay buffer
#
# # draw samples from the generator
# gen_inputs = [Z0] + [None for gm in gen_modules[1:]]
# XIZ0 = gen_network.apply(rand_vals=gen_inputs, batch_size=None)
#
# # feed real data and generated data through discriminator
# #   -- optimize with respect to discriminator output from a subset of the
# #      discriminator's modules.
# if all_disc:
#     # multi-scale discriminator guidance
#     ret_vals = range(1, len(disc_network.modules))
# else:
#     # full-scale discriminator guidance only
#     ret_vals = [ (len(disc_network.modules)-1) ]
# p_real = disc_network.apply(input=X, ret_vals=ret_vals, app_sigm=False)
# p_gen = disc_network.apply(input=XIZ0, ret_vals=ret_vals, app_sigm=False)
# p_er = disc_network.apply(input=Xer, ret_vals=ret_vals, app_sigm=False)
# print("Gathering discriminator signal from {} layers...".format(len(p_er)))
#
# # compute costs based on discriminator output for real/generated data
# d_cost_reals = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_real]
# d_cost_gens  = [bce(sigmoid(p), T.zeros(p.shape)).mean() for p in p_gen]
# d_cost_ers   = [bce(sigmoid(p), T.zeros(p.shape)).mean() for p in p_er]
# g_cost_ds    = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_gen]
# # reweight costs based on depth in discriminator (costs get heavier higher up)
# if use_weights:
#     weights = [float(i)/len(range(1,len(p_gen)+1)) for i in range(1,len(p_gen)+1)]
#     scale = sum(weights)
#     weights = [w/scale for w in weights]
# else:
#     weights = [float(1)/len(range(1,len(p_gen)+1)) for i in range(1,len(p_gen)+1)]
#     scale = sum(weights)
#     weights = [w/scale for w in weights]
# print("Discriminator signal weights {}...".format(weights))
# d_cost_real = sum([w*c for w, c in zip(weights, d_cost_reals)])
# d_cost_gen = sum([w*c for w, c in zip(weights, d_cost_gens)])
# d_cost_er = sum([w*c for w, c in zip(weights, d_cost_ers)])
# g_cost_d = sum([w*c for w, c in zip(weights, g_cost_ds)])
#
#
# # switch costs based on use of experience replay
# if use_er:
#     a1, a2 = 0.5, 0.5
# else:
#     a1, a2 = 1.0, 0.0
# d_cost = d_cost_real + a1*d_cost_gen + a2*d_cost_er + \
#          (1e-5 * sum([T.sum(p**2.0) for p in disc_params]))
# g_cost = g_cost_d + (1e-5 * sum([T.sum(p**2.0) for p in gen_params]))
#
# cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]
#
# lrt = sharedX(lr)
# d_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4, regularizer=updates.Regularizer(l2=l2))
# g_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4, regularizer=updates.Regularizer(l2=l2))
# d_updates = d_updater(disc_params, d_cost)
# g_updates = g_updater(gen_params, g_cost)
# updates = d_updates + g_updates
#
# print 'COMPILING'
# t = time()
# _train_g = theano.function([X, Z0, Xer], cost, updates=g_updates)
# _train_d = theano.function([X, Z0, Xer], cost, updates=d_updates)
# _gen = theano.function([Z0], XIZ0)
# print "{0:.2f} seconds to compile theano functions".format(time()-t)
#
# f_log = open("{}/{}.ndjson".format(log_dir, desc), 'wb')
# log_fields = [
#     'n_epochs',
#     'n_updates',
#     'n_examples',
#     'n_seconds',
#     'g_cost',
#     'd_cost',
# ]
#
# # initialize an experience replay buffer
# er_buffer = floatX(np.zeros((er_buffer_size, nc*npx*npx)))
# start_idx = 0
# end_idx = 1000
# print("Initializing experience replay buffer...")
# while start_idx < er_buffer_size:
#     samples = gen_network.generate_samples(1000)
#     samples = samples.reshape((1000,-1))
#     end_idx = min(end_idx, er_buffer_size)
#     er_buffer[start_idx:end_idx,:] = samples[:(end_idx-start_idx),:]
#     start_idx += 1000
#     end_idx += 1000
# print("DONE.")
#
# print desc.upper()
#
# log_name = "{}/RESULTS.txt".format(sample_dir)
# out_file = open(log_name, 'wb')
#
# n_updates = 0
# n_check = 0
# n_epochs = 0
# n_updates = 0
# n_examples = 0
# t = time()
# gauss_blur_weights = np.linspace(0.0, 1.0, 25) # weights for distribution "annealing"
# sample_z0mb = rand_gen(size=(200, nz0)) # noise samples for top generator module
# for epoch in range(1, niter+niter_decay+1):
#     Xtr = shuffle(Xtr)
#     g_cost = 0
#     g_cost_d = 0
#     d_cost = 0
#     d_cost_real = 0
#     gc_iter = 0
#     dc_iter = 0
#     for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
#         if epoch < gauss_blur_weights.shape[0]:
#             w_x = gauss_blur_weights[epoch]
#         else:
#             w_x = 1.0
#         w_g = 1.0 - w_x
#         if use_annealing and (w_x < 0.999):
#             imb = gauss_blur(imb, Xtr_std, w_x, w_g)
#         imb = train_transform(imb)
#         z0mb = rand_gen(size=(len(imb), nz0))
#         if n_updates % (k+1) == 0:
#             # sample data from experience replay buffer
#             xer = train_transform(sample_exprep_buffer(er_buffer, len(imb)))
#             # compute generator cost and apply update
#             result = _train_g(imb, z0mb, xer)
#             g_cost += result[0]
#             g_cost_d += result[2]
#             gc_iter += 1
#         else:
#             # sample data from experience replay buffer
#             xer = train_transform(sample_exprep_buffer(er_buffer, len(imb)))
#             # compute discriminator cost and apply update
#             result = _train_d(imb, z0mb, xer)
#             d_cost += result[1]
#             d_cost_real += result[3]
#             dc_iter += 1
#         n_updates += 1
#         n_examples += len(imb)
#         # update experience replay buffer (a better update schedule may be helpful)
#         if ((n_updates % (min(25,epoch)*25)) == 0) and use_er:
#             update_exprep_buffer(er_buffer, gen_network, replace_frac=0.10)
#     str1 = "Epoch {}:".format(epoch)
#     str2 = "    g_cost: {0:.4f},      d_cost: {1:.4f}".format((g_cost/gc_iter),(d_cost/dc_iter))
#     str3 = "  g_cost_d: {0:.4f}, d_cost_real: {1:.4f}".format((g_cost_d/gc_iter),(d_cost_real/dc_iter))
#     joint_str = "\n".join([str1, str2, str3])
#     print(joint_str)
#     out_file.write(joint_str+"\n")
#     out_file.flush()
#     # generate some samples from the model, for visualization
#     samples = np.asarray(_gen(sample_z0mb))
#     color_grid_vis(draw_transform(samples), (10, 20), "{}/{}.png".format(sample_dir, n_epochs))
#     n_epochs += 1
#     if n_epochs > niter:
#         lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
#     if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
#         joblib.dump([p.get_value() for p in gen_params], "{}/{}_gen_params.jl".format(model_dir, n_epochs))
#         joblib.dump([p.get_value() for p in disc_params], "{}/{}_disc_params.jl".format(model_dir, n_epochs))
