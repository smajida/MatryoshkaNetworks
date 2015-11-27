import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import shuffle, iter_data
from load import mnist_with_valid_set

#
# Phil's business
#
from MatryoshkaModules import DiscConvModule, DiscFCModule, GenConvModule, \
                              GenFCModule, BasicConvModule

# path for dumping experiment info and fetching dataset
EXP_DIR = "./mnist"

trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set("{}/data".format(EXP_DIR))

vaX = floatX(vaX)/255.

k = 1             # # of discrim updates for each gen update
l2 = 1.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz0 = 32          # # of dim for Z0
nz1 = 8           # # of dim for Z1
ngfc = 256        # # of gen units for fully connected layers
ndfc = 256        # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain, nval, ntest = len(trX), len(vaX), len(teX)

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X

def rand_gen(size):
    #r_vals = floatX(np_rng.uniform(-1., 1., size=size))
    r_vals = floatX(np_rng.normal(size=size))
    return r_vals


desc = 'matronet_1'
model_dir = "{}/models/{}".format(EXP_DIR, desc)
sample_dir = "{}/samples/{}".format(EXP_DIR, desc)
log_dir = "{}/logs".format(EXP_DIR)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

#
# Define some modules to use in the generator
#
gen_module_1 = \
GenFCModule(
    rand_dim=nz0,
    out_dim=(ngf*2*7*7),
    fc_dim=ngfc,
    apply_bn_1=True,
    apply_bn_2=True,
    init_func=gifn,
    rand_type='normal',
    mod_name='gen_mod_1'
)

gen_module_2 = \
GenConvModule(
    filt_shape=(3,3),
    in_chans=(ngf*2),
    out_chans=ngf,
    rand_chans=nz1,
    apply_bn_1=True,
    apply_bn_2=True,
    us_stride=2,
    init_func=gifn,
    use_rand=True,
    use_pooling=False,
    rand_type='normal',
    mod_name='gen_mod_2'
)

gen_module_3 = \
GenConvModule(
    filt_shape=(3,3),
    in_chans=ngf,
    out_chans=ngf,
    rand_chans=nz1,
    apply_bn_1=True,
    apply_bn_2=True,
    us_stride=2,
    init_func=gifn,
    use_rand=True,
    use_pooling=False,
    rand_type='normal',
    mod_name='gen_mod_3'
)

gwx = gifn((nc, ngf, 3, 3), 'gwx')

#
# Define some modules to use in the discriminator
#
disc_module_1 = \
DiscConvModule(
    filt_shape=(3,3),
    in_chans=nc,
    out_chans=ndf,
    apply_bn_1=False,
    apply_bn_2=True,
    ds_stride=2,
    use_pooling=False,
    init_func=difn,
    mod_name='disc_mod_1'
)

disc_module_2 = \
DiscConvModule(
    filt_shape=(3,3),
    in_chans=ndf,
    out_chans=(ndf*2),
    apply_bn_1=True,
    apply_bn_2=True,
    ds_stride=2,
    use_pooling=False,
    init_func=difn,
    mod_name='disc_mod_2'
)

disc_module_3 = \
DiscFCModule(
    fc_dim=ndfc,
    in_dim=(ndf*2*7*7),
    apply_bn=True,
    init_func=difn,
    mod_name='disc_mod_3'
)
 
#
# Grab parameters from generator and discriminator
#
gen_params = gen_module_1.params + \
             gen_module_2.params + \
             gen_module_3.params + \
             [gwx]

discrim_params = disc_module_1.params + \
                 disc_module_2.params + \
                 disc_module_3.params

def gen(Z0, wx):
    # feedforward through the fully connected part of generator
    h2 = gen_module_1.apply(rand_vals=Z0)
    # reshape as input to a conv layer (in 7x7 grid)
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    # feedforward through convolutional generator module
    h3 = gen_module_2.apply(h2, rand_vals=None)
    # feedforward through convolutional generator module
    h4 = gen_module_3.apply(h3, rand_vals=None)
    # feedforward through another conv and clamp to [0,1]
    h5 = dnn_conv(h4, wx, subsample=(1, 1), border_mode=(1, 1))
    x = sigmoid(h5)
    return x

def discrim(X):
    # apply 3x3 double conv discriminator module
    h1, y1 = disc_module_1.apply(X)
    # apply 3x3 double conv discriminator module
    h2, y2 = disc_module_2.apply(h1)
    # concat label info and feedforward through fc module
    h2 = T.flatten(h2, 2)
    y3 = disc_module_3.apply(h2)
    return y1, y2, y3

X = T.tensor4()
Z0 = T.matrix()

# draw samples from the generator
gX = gen(Z0, gwx)

# feed real data and generated data through discriminator
p_real = discrim(X)
p_gen = discrim(gX)

# compute costs based on discriminator output for real/generated data
d_cost_real = sum([bce(p, T.ones(p.shape)).mean() for p in p_real])
d_cost_gen = sum([bce(p, T.zeros(p.shape)).mean() for p in p_gen])
g_cost_d = sum([bce(p, T.ones(p.shape)).mean() for p in p_gen])

#d_cost_real = bce(p_real[-1], T.ones(p_real[-1].shape)).mean()
#d_cost_gen = bce(p_gen[-1], T.zeros(p_gen[-1].shape)).mean()
#g_cost_d = bce(p_gen[-1], T.ones(p_gen[-1].shape)).mean()

d_cost = d_cost_real + d_cost_gen + (1e-5 * sum([T.sum(p**2.0) for p in discrim_params]))
g_cost = g_cost_d + (1e-5 * sum([T.sum(p**2.0) for p in gen_params]))

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z0], cost, updates=g_updates)
_train_d = theano.function([X, Z0], cost, updates=d_updates)
_gen = theano.function([Z0], gX)
print '%.2f seconds to compile theano functions'%(time()-t)


f_log = open("{}/{}.ndjson".format(log_dir, desc), 'wb')
log_fields = [
    'n_epochs', 
    'n_updates', 
    'n_examples', 
    'n_seconds',
    'g_cost',
    'd_cost',
]

print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
sample_z0mb = rand_gen(size=(200, nz0)) # noise samples for top generator module
for epoch in range(1, niter+niter_decay+1):
    trX = shuffle(trX)
    for imb in tqdm(iter_data(trX, size=nbatch), total=ntrain/nbatch):
        imb = transform(imb)
        z0mb = rand_gen(size=(len(imb), nz0))
        if n_updates % (k+1) == 0:
            cost = _train_g(imb, z0mb)
        else:
            cost = _train_d(imb, z0mb)
        n_updates += 1
        n_examples += len(imb)
    samples = np.asarray(_gen(sample_z0mb))
    grayscale_grid_vis(inverse_transform(samples), (10, 20), "{}/{}.png".format(sample_dir, n_epochs))
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
    if n_epochs in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        joblib.dump([p.get_value() for p in gen_params], "{}/{}_gen_params.jl".format(model_dir, n_epochs))
        joblib.dump([p.get_value() for p in discrim_params], "{}/{}_discrim_params.jl".format(model_dir, n_epochs))