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
from MatryoshkaModules import DiscFCModule, GenTopModule, \
                              BasicConvModule, GenConvResModule, \
                              DiscConvResModule
from MatryoshkaNetworks import GenNetworkGAN, DiscNetworkGAN, VarInfModel



# path for dumping experiment info and fetching dataset
EXP_DIR = "./faces_celeba"

# setup paths for dumping diagnostic info
desc = 'test_dcgan_paper_model'
result_dir = "{}/results/{}".format(EXP_DIR, desc)
gen_param_file = "{}/gen_params.pkl".format(result_dir)
disc_param_file = "{}/disc_params.pkl".format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# locations of 64x64 faces dataset -- stored as a collection of .npy files
data_dir = "{}/data".format(EXP_DIR)
# get a list of the .npy files that contain images in this directory. there
# shouldn't be any other files in the directory (hackish, but easy).
data_files = os.listdir(data_dir)
data_files.sort()
data_files = ["{}/{}".format(data_dir, file_name) for file_name in data_files]

def scale_to_tanh_range(X):
    """
    Scale the given 2d array to be in tanh range (i.e. -1...1).
    """
    X = (X / 127.5) - 1.0
    X_std = np.std(X, axis=0, keepdims=True)
    return X, X_std

def load_and_scale_data(npy_file_name):
    """
    Load and scale data from the given npy file, and compute standard deviation
    too, to use when doing distribution annealing.
    """
    np_ary = np.load(npy_file_name)
    np_ary = np_ary.astype(theano.config.floatX)
    X, X_std = scale_to_tanh_range(np_ary)
    return X, X_std


set_seed(1)     # seed for shared rngs
k = 1            # # of discrim updates for each gen update
b1 = 0.5         # momentum term of adam
nc = 3           # # of channels in image
nbatch = 100     # # of examples in batch
npx = 64         # # of pixels width/height of images
nz0 = 128         # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngfc = 256        # # of gen units for fully connected layers
ndfc = 256        # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.00015       # initial learning rate for adam
slow_buffer_size = 200000  # size of slow replay buffer
fast_buffer_size = 20000   # size of fast replay buffer


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

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

################################
# Setup the generator network #
###############################

gen_module_1 = \
GenTopModule(
    rand_dim=nz0,
    out_shape=(ngf*8, 4, 4),
    fc_dim=ngfc,
    use_fc=False,
    apply_bn=True,
    mod_name='gen_mod_1'
) # output is (batch, ngf*8, 4, 4)

gen_module_2 = \
GenConvResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=False,
    use_conv=False,
    us_stride=2,
    mod_name='gen_mod_2'
) # output is (batch, ngf*4, 8, 8)

gen_module_3 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=False,
    use_conv=False,
    us_stride=2,
    mod_name='gen_mod_3'
) # output is (batch, ngf*2, 16, 16)

gen_module_4 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=False,
    use_conv=False,
    us_stride=2,
    mod_name='gen_mod_4'
) # output is (batch, ngf*2, 32, 32)

gen_module_5 = \
GenConvResModule(
    in_chans=(ngf*1),
    out_chans=40,
    conv_chans=40,
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=False,
    use_conv=False,
    us_stride=2,
    mod_name='gen_mod_5'
) # output is (batch, ngf*1, 64, 64)

gen_module_6 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=40,
    out_chans=nc,
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='gen_mod_6'
) # output is (batch, c, 64, 64)

gen_modules = [gen_module_1, gen_module_2, gen_module_3,
               gen_module_4, gen_module_5, gen_module_6]

# Initialize the generator network
gen_network = GenNetworkGAN(modules=gen_modules, output_transform=tanh)
# load params from pre-trained network
gen_network.load_params(gen_param_file)


###################################
# Setup the discriminator network #
###################################

disc_module_1 = \
BasicConvModule(
    filt_shape=(5,5),
    in_chans=nc,
    out_chans=(ndf*1),
    apply_bn=False,
    stride='double',
    act_func='lrelu',
    mod_name='disc_mod_1'
) # output is (batch, ndf*1, 32, 32)

disc_module_2 = \
DiscConvResModule(
    in_chans=(ndf*1),
    out_chans=(ndf*2),
    conv_chans=ndf,
    filt_shape=(5,5),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_2'
) # output is (batch, ndf*2, 16, 16)

disc_module_3 = \
DiscConvResModule(
    in_chans=(ndf*2),
    out_chans=(ndf*4),
    conv_chans=ndf,
    filt_shape=(5,5),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_3'
) # output is (batch, ndf*4, 8, 8)

disc_module_4 = \
DiscConvResModule(
    in_chans=(ndf*4),
    out_chans=(ndf*8),
    conv_chans=ndf,
    filt_shape=(5,5),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_4'
) # output is (batch, ndf*8, 4, 4)

disc_module_5 = \
DiscFCModule(
    fc_dim=ndfc,
    in_dim=(ndf*8*4*4),
    use_fc=False,
    apply_bn=True,
    unif_drop=0.0,
    mod_name='disc_mod_5'
) # output is (batch, 1)

disc_modules = [disc_module_1, disc_module_2, disc_module_3,
                disc_module_4, disc_module_5]

# Initialize the discriminator network
disc_network = DiscNetworkGAN(modules=disc_modules)
disc_params = disc_network.params

###########################################
# Construct a VarInfModel for gen_network #
###########################################
Xtr, Xtr_std = load_and_scale_data(data_files[0])
print("data_files[0]: {}".format(data_files[0]))

print("Xtr.shape: {}".format(Xtr.shape))

Xtr_rec = Xtr[0:100,:]
Mtr_rec = floatX(np.ones(Xtr_rec.shape))
print("Building VarInfModel...")
VIM = VarInfModel(Xtr_rec, Mtr_rec, gen_network, post_logvar=-4.0)
print("Testing VarInfModel...")
opt_cost, vfe_bounds = VIM.train(0.001)
vfe_bounds = VIM.sample_vfe_bounds()
test_recons = VIM.sample_Xg()
color_grid_vis(draw_transform(Xtr_rec), (10, 10), "{}/Xtr_rec.png".format(result_dir))

####################################
# Setup the optimization objective #
####################################

X = T.tensor4()   # symbolic var for real inputs to discriminator
Z0 = T.matrix()   # symbolic var for rand values to pass into generator

# draw samples from the generator
gen_inputs = [Z0] + [None for gm in gen_modules[1:]]
XIZ0 = gen_network.apply(rand_vals=gen_inputs, batch_size=None)

# feed real data and generated data through discriminator
#   -- optimize with respect to discriminator output from a subset of the
#      discriminator's modules.
ret_vals = range(1, len(disc_network.modules))
p_real = disc_network.apply(input=X, ret_vals=ret_vals, app_sigm=False)
p_gen = disc_network.apply(input=XIZ0, ret_vals=ret_vals, app_sigm=False)
print("Gathering discriminator signal from {} layers...".format(len(p_real)))

# compute costs based on discriminator output for real/generated data
d_cost_obs = sum([T.mean(bce(sigmoid(p), T.ones(p.shape)), axis=1) for p in p_real])
d_cost_reals = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_real]
d_cost_gens  = [bce(sigmoid(p), T.zeros(p.shape)).mean() for p in p_gen]
g_cost_ds    = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_gen]
# reweight costs based on depth in discriminator (costs get heavier higher up)
d_weights = [1.0 for i in range(1,len(p_gen)+1)]
d_weights[0] = 0.0
scale = sum(d_weights)
d_weights = [w/scale for w in d_weights]
print("Discriminator signal weights {}...".format(d_weights))
d_cost_real = sum([w*c for w, c in zip(d_weights, d_cost_reals)])
d_cost_gen = sum([w*c for w, c in zip(d_weights, d_cost_gens)])

# switch costs based on use of experience replay
d_cost = d_cost_real + d_cost_gen + \
         (3e-5 * sum([T.sum(p**2.0) for p in disc_params]))

all_costs = [d_cost, d_cost_real, d_cost_gen] + g_cost_ds

lrd = sharedX(lr)
d_updater = updates.Adam(lr=lrd, b1=b1, b2=0.98, e=1e-4)
d_updates = d_updater(disc_params, d_cost)

print 'COMPILING'
t = time()
_train_d = theano.function([X, Z0], all_costs, updates=d_updates)
_gen = theano.function([Z0], XIZ0)
_disc = theano.function([X], d_cost_obs)
print "{0:.2f} seconds to compile theano functions".format(time()-t)
# test disc cost func
temp = _disc(train_transform(Xtr[0:50,:]))
print("temp.shape: {}".format(temp.shape))

print desc.upper()

log_name = "{}/EVAL.txt".format(result_dir)
out_file = open(log_name, 'wb')

t = time()
sample_z0mb = np.repeat(rand_gen(size=(10, nz0)), 20, axis=0)
for epoch in range(1, niter+niter_decay+1):
    # load a file containing a subset of the large full training set
    Xtr, Xtr_std = load_and_scale_data(data_files[epoch % len(data_files)])
    Xtr = shuffle(Xtr)
    ntrain = Xtr.shape[0]
    # initialize cost recording arrays
    d_costs = [0. for c in all_costs]
    dc_iter = 0
    rec_iter = 0
    rec_cost = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        imb = train_transform(imb)
        z0mb = rand_gen(size=(len(imb), nz0))
        # compute discriminator cost and apply update
        result = _train_d(imb, z0mb)
        d_costs = [(v1 + v2) for v1, v2 in zip(d_costs, result)]
        dc_iter += 1
        # train the bootleg variational inference model
        opt_cost, vfe_bounds = VIM.train(0.001)
        rec_cost += opt_cost
        rec_iter += 1
    ############################
    # QUANTITATIVE DIAGNOSTICS #
    ############################
    d_costs = [(v / dc_iter) for v in d_costs]
    rec_cost = rec_cost / rec_iter
    str1 = "Epoch {}:".format(epoch)
    str2 = "    d_cost: {1:.4f}, d_cost_real: {2:.4f}, d_cost_gen: {3:.4f}, rec_cost: {4:.4f}".format( \
            d_costs[0], d_costs[1], d_costs[2], rec_cost)
    str3 = "    -- g_cost_d_train: {}".format( \
            ", ".join(["{0:d}: {1:.2f}".format(j,c) for j, c in enumerate(d_costs[3:])]))
    joint_str = "\n".join([str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    # generate some samples from the model, for visualization
    samples = floatX( _gen(sample_z0mb) )
    d_cost_samps = _disc(samples)
    #sort_idx = np.argsort(-1.0 * d_cost_samps)
    #samples = samples[sort_idx,:,:,:]
    color_grid_vis(draw_transform(samples), (10, 20), "{}/eval_gen_{}.png".format(result_dir, epoch))
    test_recons = VIM.sample_Xg()
    color_grid_vis(draw_transform(test_recons), (10, 20), "{}/eval_rec_{}.png".format(result_dir, epoch))






##############
# EYE BUFFER #
##############
