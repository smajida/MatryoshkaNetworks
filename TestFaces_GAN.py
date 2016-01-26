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
from MatryoshkaModules import DiscFCModule, GenFCModule, \
                              BasicConvModule, GenConvResModule, \
                              DiscConvResModule
from MatryoshkaNetworks import GenNetworkGAN, DiscNetworkGAN, VarInfModel



# path for dumping experiment info and fetching dataset
EXP_DIR = "./faces_celeba"
DATA_SIZE = 250000

# setup paths for dumping diagnostic info
desc = 'test_gan_big_model'
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
nbatch = 64      # # of examples in batch
npx = 64         # # of pixels width/height of images
nz0 = 128         # # of dim for Z0
nz1 = 16          # # of dim for Z1
ngfc = 256        # # of gen units for fully connected layers
ndfc = 256        # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 150       # # of iter at starting learning rate
niter_decay = 200 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
er_buffer_size = DATA_SIZE # size of "experience replay" buffer
multi_rand = True   # whether to use stochastic variables at multiple scales
multi_disc = True   # whether to use discriminator guidance at multiple scales
use_er = True     # whether to use experience replay
use_conv = True   # whether to use "internal" conv layers in gen/disc networks
use_annealing = True # whether to use "annealing" of the target distribution


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

def mixed_hinge_loss(vals):
    """
    Compute mixed L1/L2 hinge loss, with hinge at 0.
    """
    clip_vals = T.maximum(vals, 0.0)
    loss_vals = 0.5*clip_vals + 0.5*clip_vals**2.0
    return loss_vals

def update_exprep_buffer(er_buffer, generator, replace_frac=0.1, do_print=False):
    """
    Update the "experience replay buffer" er_buffer using samples generated by
    generator. Replace replace_frac of buffer with new samples.

    Assume er_buffer is a 2d numpy array.
    """
    buffer_size = er_buffer.shape[0]
    new_sample_count = int(buffer_size * replace_frac)
    new_samples = floatX(np.zeros((new_sample_count, nc*npx*npx)))
    start_idx = 0
    end_idx = 100
    if do_print:
        print("Updating experience replay buffer...")
    while start_idx < new_sample_count:
        samples = generator.generate_samples(100)
        samples = samples.reshape((100,-1))
        end_idx = min(end_idx, new_sample_count)
        new_samples[start_idx:end_idx,:] = samples[:(end_idx-start_idx),:]
        start_idx += 100
        end_idx += 100
    idx = np.arange(buffer_size)
    npr.shuffle(idx)
    replace_idx = idx[:new_sample_count]
    er_buffer[replace_idx,:] = new_samples
    return er_buffer

def sample_exprep_buffer(er_buffer, sample_count):
    """
    Sample from the "experience replay buffer" er_buffer, with replacement.
    """
    buffer_size = er_buffer.shape[0]
    idx = npr.randint(0,high=buffer_size)
    samples = er_buffer[idx,:]
    return samples

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

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

###############################
# Setup the generator network #
###############################

gen_module_1 = \
GenFCModule(
    rand_dim=nz0,
    out_shape=(ngf*8, 2, 2),
    fc_dim=ngfc,
    use_fc=True,
    apply_bn=True,
    mod_name='gen_mod_1'
) # output is (batch, ngf*8, 2, 2)

gen_module_2 = \
GenConvResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*8),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='gen_mod_2'
) # output is (batch, ngf*8, 4, 4)

gen_module_3 = \
GenConvResModule(
    in_chans=(ngf*8),
    out_chans=(ngf*4),
    conv_chans=(ngf*4),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='gen_mod_3'
) # output is (batch, ngf*4, 8, 8)

gen_module_4 = \
GenConvResModule(
    in_chans=(ngf*4),
    out_chans=(ngf*2),
    conv_chans=(ngf*2),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='gen_mod_4'
) # output is (batch, ngf*2, 16, 16)

gen_module_5 = \
GenConvResModule(
    in_chans=(ngf*2),
    out_chans=(ngf*1),
    conv_chans=(ngf*1),
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='gen_mod_5'
) # output is (batch, ngf*2, 32, 32)

gen_module_6 = \
GenConvResModule(
    in_chans=(ngf*1),
    out_chans=32,
    conv_chans=32,
    filt_shape=(3,3),
    rand_chans=nz1,
    use_rand=multi_rand,
    use_conv=use_conv,
    us_stride=2,
    mod_name='gen_mod_6'
) # output is (batch, ngf*1, 64, 64)

gen_module_7 = \
BasicConvModule(
    filt_shape=(3,3),
    in_chans=32,
    out_chans=nc,
    apply_bn=False,
    stride='single',
    act_func='ident',
    mod_name='gen_mod_7'
) # output is (batch, c, 64, 64)

gen_modules = [gen_module_1, gen_module_2, gen_module_3, gen_module_4,
               gen_module_5, gen_module_6, gen_module_7]

# Initialize the generator network
gen_network = GenNetworkGAN(modules=gen_modules, output_transform=tanh)
gen_params = gen_network.params


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
DiscConvResModule(
    in_chans=(ndf*8),
    out_chans=(ndf*8),
    conv_chans=ndf,
    filt_shape=(3,3),
    use_conv=False,
    ds_stride=2,
    mod_name='disc_mod_5'
) # output is (batch, ndf*8, 2, 2)

disc_module_6 = \
DiscFCModule(
    fc_dim=ndfc,
    in_dim=(ndf*8*2*2),
    use_fc=False,
    apply_bn=True,
    mod_name='disc_mod_6'
) # output is (batch, 1)

disc_modules = [disc_module_1, disc_module_2, disc_module_3,
                disc_module_4, disc_module_5, disc_module_6]

# Initialize the discriminator network
disc_network = DiscNetworkGAN(modules=disc_modules)
disc_params = disc_network.params

###########################################
# Construct a VarInfModel for gen_network #
###########################################
Xtr, Xtr_std = load_and_scale_data(data_files[0])
print("data_files[0]: {}".format(data_files[0]))

print("Xtr.shape: {}".format(Xtr.shape))

Xtr_rec = Xtr[0:200,:]
Mtr_rec = floatX(np.ones(Xtr_rec.shape))
print("Building VarInfModel...")
VIM = VarInfModel(Xtr_rec, Mtr_rec, gen_network, post_logvar=-4.0)
print("Testing VarInfModel...")
opt_cost, vfe_bounds = VIM.train(0.001)
vfe_bounds = VIM.sample_vfe_bounds()
test_recons = VIM.sample_Xg()
color_grid_vis(draw_transform(Xtr_rec), (10, 20), "{}/Xtr_rec.png".format(result_dir))


####################################
# Setup the optimization objective #
####################################

X = T.tensor4()   # symbolic var for real inputs to discriminator
Z0 = T.matrix()   # symbolic var for rand values to pass into generator
Xer = T.tensor4() # symbolic var for samples from experience replay buffer

# draw samples from the generator
gen_inputs = [Z0] + [None for gm in gen_modules[1:]]
XIZ0 = gen_network.apply(rand_vals=gen_inputs, batch_size=None)

# feed real data and generated data through discriminator
#   -- optimize with respect to discriminator output from a subset of the
#      discriminator's modules.
if multi_disc:
    # multi-scale discriminator guidance
    ret_vals = range(1, len(disc_network.modules))
else:
    # full-scale discriminator guidance only
    ret_vals = [ (len(disc_network.modules)-1) ]
p_real = disc_network.apply(input=X, ret_vals=ret_vals, app_sigm=False)
p_gen = disc_network.apply(input=XIZ0, ret_vals=ret_vals, app_sigm=False)
p_er = disc_network.apply(input=Xer, ret_vals=ret_vals, app_sigm=False)
print("Gathering discriminator signal from {} layers...".format(len(p_er)))

# compute costs based on discriminator output for real/generated data
d_cost_reals = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_real]
d_cost_gens  = [bce(sigmoid(p), T.zeros(p.shape)).mean() for p in p_gen]
d_cost_ers   = [bce(sigmoid(p), T.zeros(p.shape)).mean() for p in p_er]
g_cost_ds    = [bce(sigmoid(p), T.ones(p.shape)).mean() for p in p_gen]
g_cost_hs    = [T.maximum((0.2-p), 0.0).mean() for p in p_gen]
# reweight costs based on depth in discriminator (costs get heavier higher up)
weights = [1.0 for i in range(1,len(p_gen)+1)]
weights[0] = 0.0
scale = sum(weights)
weights = [w/scale for w in weights]
print("Discriminator signal weights {}...".format(weights))
d_cost_real = sum([w*c for w, c in zip(weights, d_cost_reals)])
d_cost_gen = sum([w*c for w, c in zip(weights, d_cost_gens)])
d_cost_er = sum([w*c for w, c in zip(weights, d_cost_ers)])
g_cost_d = sum([w*c for w, c in zip(weights, g_cost_ds)])
g_cost_h = sum([w*c for w, c in zip(weights, g_cost_hs)])


# switch costs based on use of experience replay
if use_er:
    a1, a2 = 0.5, 0.5
else:
    a1, a2 = 1.0, 0.0
d_cost = d_cost_real + a1*d_cost_gen + a2*d_cost_er + \
         (4e-5 * sum([T.sum(p**2.0) for p in disc_params]))
g_cost = g_cost_h + (1e-5 * sum([T.sum(p**2.0) for p in gen_params]))

all_costs = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen, d_cost_er] + g_cost_ds

lrt = sharedX(lr)
lrd = sharedX(lr/2.0)
d_updater = updates.Adam(lr=lrd, b1=b1, b2=0.98, e=1e-4)
g_updater = updates.Adam(lr=lrt, b1=b1, b2=0.98, e=1e-4)
d_updates = d_updater(disc_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z0, Xer], all_costs, updates=g_updates)
_train_d = theano.function([X, Z0, Xer], all_costs, updates=d_updates)
_gen = theano.function([Z0], XIZ0)
print "{0:.2f} seconds to compile theano functions".format(time()-t)


# initialize an experience replay buffer
er_buffer = floatX(np.zeros((er_buffer_size, nc*npx*npx)))
start_idx = 0
end_idx = 200
print("Initializing experience replay buffer...")
while start_idx < er_buffer_size:
    samples = gen_network.generate_samples(200)
    samples = samples.reshape((200,-1))
    end_idx = min(end_idx, er_buffer_size)
    er_buffer[start_idx:end_idx,:] = samples[:(end_idx-start_idx),:]
    start_idx += 200
    end_idx += 200
print("DONE.")

print desc.upper()

log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')

n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
gauss_blur_weights = np.linspace(0.0, 1.0, 50) # weights for distribution "annealing"
sample_z0mb = rand_gen(size=(200, nz0)) # noise samples for top generator module
for epoch in range(1, niter+niter_decay+1):
    # load a file containing a subset of the large full training set
    Xtr, Xtr_std = load_and_scale_data(data_files[epoch % len(data_files)])
    Xtr = shuffle(Xtr)
    ntrain = Xtr.shape[0]
    g_costs = [0. for c in all_costs]
    d_costs = [0. for c in all_costs]
    gc_iter = 0
    dc_iter = 0
    rec_iter = 0
    rec_cost = 0.
    for imb in tqdm(iter_data(Xtr, size=nbatch), total=ntrain/nbatch):
        if epoch < gauss_blur_weights.shape[0]:
            w_x = gauss_blur_weights[epoch]
        else:
            w_x = 1.0
        w_g = 1.0 - w_x
        if use_annealing and (w_x < 0.999):
            imb = np.clip(gauss_blur(imb, Xtr_std, w_x, w_g),
                          a_min=-1.0, a_max=1.0)
        imb = train_transform(imb)
        z0mb = rand_gen(size=(len(imb), nz0))
        if n_updates % (k+1) == 0:
            # sample data from experience replay buffer
            xer = train_transform(sample_exprep_buffer(er_buffer, len(imb)))
            # compute generator cost and apply update
            result = _train_g(imb, z0mb, xer)
            g_costs = [(v1 + v2) for v1, v2 in zip(g_costs, result)]
            gc_iter += 1
        else:
            # sample data from experience replay buffer
            xer = train_transform(sample_exprep_buffer(er_buffer, len(imb)))
            # compute discriminator cost and apply update
            result = _train_d(imb, z0mb, xer)
            d_costs = [(v1 + v2) for v1, v2 in zip(d_costs, result)]
            dc_iter += 1
        if ((n_updates % 10) == 0):
            # train the bootleg variational inference model
            opt_cost, vfe_bounds = VIM.train(0.001)
            rec_cost += opt_cost
            rec_iter += 1
        n_updates += 1
        n_examples += len(imb)
        # update experience replay buffer (a better update schedule may be helpful)
        if ((n_updates % (min(10,epoch)*15)) == 0) and use_er:
            update_exprep_buffer(er_buffer, gen_network, replace_frac=0.10)
    ###################
    # SAVE PARAMETERS #
    ###################
    gen_network.dump_params(gen_param_file)
    disc_network.dump_params(disc_param_file)
    ############################
    # QUANTITATIVE DIAGNOSTICS #
    ############################
    g_costs = [(v / gc_iter) for v in g_costs]
    d_costs = [(v / dc_iter) for v in d_costs]
    rec_cost = rec_cost / rec_iter
    str1 = "Epoch {}:".format(epoch)
    str2 = "    g_cost: {0:.4f}, d_cost: {1:.4f}, rec_cost: {2:.4f}".format( \
            g_costs[0], d_costs[1], rec_cost)
    str3 = "    -- g_cost_d: {0:.4f}, d_cost_real: {1:.4f}, d_cost_gen: {2:.4f}, d_cost_er: {3:.4f}".format( \
            g_costs[2], d_costs[3], d_costs[4], d_costs[5])
    str4 = "    -- g_cost_ds: {}".format( \
            ", ".join(["{0:d}: {1:.2f}".format(j,c) for j, c in enumerate(d_costs[6:])]))
    joint_str = "\n".join([str1, str2, str3, str4])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    n_epochs += 1
    # generate some samples from the model, for visualization
    samples = np.asarray(_gen(sample_z0mb))
    color_grid_vis(draw_transform(samples), (10, 20), "{}/gen_{}.png".format(result_dir, n_epochs))
    test_recons = VIM.sample_Xg()
    color_grid_vis(draw_transform(test_recons), (10, 20), "{}/rec_{}.png".format(result_dir, n_epochs))
    if (n_epochs == 50) or (n_epochs == 100):
        old_lr = lrt.get_value(borrow=False)
        new_lr = 0.5 * old_lr
        lrt.set_value(floatX(new_lr))
        lrd.set_value(floatX(new_lr/2.0))
    if n_epochs > niter:
        # reduce learning rate and keep discriminator at half rate
        iters_left = (niter_decay + niter) - n_epochs + 1
        old_lr = lrt.get_value(borrow=False)
        new_lr = old_lr - (old_lr / iters_left)
        lrt.set_value(floatX(new_lr))
        lrd.set_value(floatX(new_lr/2.0))





##############
# EYE BUFFER #
##############
