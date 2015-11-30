import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy


#############################
# BASIC CONVOLUTIONAL LAYER #
#############################

class BasicConvModule(object):
    """
    Simple convolutional layer for use anywhere?

    Params:
        filt_shape: filter shape, should be square and odd dim
        in_chans: number of channels in input
        out_chans: number of channels to produce as output
        apply_bn: whether to apply batch normalization after conv
        act_func: should be "ident", "relu", or "lrelu"
        init_func: function for initializing module parameters
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans,
                 apply_bn=True, act_func='ident', init_func=None,
                 mod_name='basic_conv'):
        assert ((filt_shape[0] % 2) > 0), "filter dim should be odd (not even)"
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.apply_bn = apply_bn
        self.act_func = act_func
        self.mod_name = mod_name
        if init_func is None:
            self.init_func = inits.Normal(scale=0.02)
        else:
            self.init_func = init_func
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        self.w1 = self.init_func((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                                 "{}_w1".format(self.mod_name))
        self.params = [self.w1]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn:
            gain_ifn = inits.Normal(loc=1., scale=0.02)
            bias_ifn = inits.Constant(c=0.)
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])
        return

    def apply(self, input, **kwargs):
        """
        Apply this convolutional module to the given input.
        """
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
        # apply first conv layer
        h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
        if self.act_func == 'ident':
            pass # apply identity activation function...
        elif self.act_func == 'lrelu':
            h1 = lrelu(h1)
        elif self.act_func == 'relu':
            h1 = relu(h1)
        else:
            assert False, "unsupported activation function."
        return h1

#############################################
# DISCRIMINATOR DOUBLE CONVOLUTIONAL MODULE #
#############################################

class DiscConvModule(object):
    """
    Module that does one layer of convolution with stride 1 followed by
    another layer of convlution with adjustable stride.

    Following the second layer of convolution, an additional convolution
    is performed that produces a single "discriminator" channel.

    Params:
        filt_shape: filter shape, should be square and odd dim
        in_chans: number of channels in input
        out_chans: number of channels to produce as output
        num_layers: number of conv layers in module -- must be 1 or 2
        apply_bn_1: whether to apply batch normalization after first conv
        apply_bn_2: whether to apply batch normalization after second conv
        ds_stride: "downsampling" stride for the second convolution
        use_pooling: whether to use max pooling or multi-striding
        init_func: function for initializing module parameters
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans, num_layers=2,
                 apply_bn_1=True, apply_bn_2=True, ds_stride=2,
                 use_pooling=True, init_func=None, mod_name='dm_conv'):
        assert ((filt_shape[0] % 2) > 0), \
                "filter dim should be odd (not even)"
        assert ((num_layers == 1) or (num_layers == 2)), \
                "num_layers must be 1 or 2."
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_layers = num_layers
        self.apply_bn_1 = apply_bn_1
        self.apply_bn_2 = apply_bn_2
        self.ds_stride = ds_stride
        self.use_pooling = use_pooling
        self.mod_name = mod_name
        if init_func is None:
            self.init_func = inits.Normal(scale=0.02)
        else:
            self.init_func = init_func
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        # initialize params for first layer and discriminator layer
        self.w1 = self.init_func((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                                 "{}_w1".format(self.mod_name))
        self.wd = self.init_func((1, self.out_chans, self.filt_dim, self.filt_dim),
                                 "{}_wd".format(self.mod_name))
        self.params = [self.w1, self.wd]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            gain_ifn = inits.Normal(loc=1., scale=0.02)
            bias_ifn = inits.Constant(c=0.)
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])

        if self.num_layers == 2:
            # initialize parameters for second layer
            self.w2 = self.init_func((self.out_chans, self.out_chans, self.filt_dim, self.filt_dim),
                                     "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
                gain_ifn = inits.Normal(loc=1., scale=0.02)
                bias_ifn = inits.Constant(c=0.)
                self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
                self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
                self.params.extend([self.g2, self.b2])
        return

    def apply(self, input, noise_sigma=None):
        """
        Apply this discriminator module to the given input. This produces a
        collection of filter responses for feedforward and a spatial grid of
        discriminator outputs.
        """
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
        ss = self.ds_stride               # stride for "learned downsampling"
        if self.num_layers == 1:
            # apply first conv layer (may include downsampling)
            if self.use_pooling:
                # change spatial dim via max pooling
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
                if self.apply_bn_1:
                    h1 = batchnorm(h1, g=self.g1, b=self.b1, n=noise_sigma)
                h1 = lrelu(h1)
                h1 = dnn_pool(h1, (ss,ss), stride=(ss, ss), mode='max', pad=(0, 0))
            else:
                # change spatial dim via strided convolution
                h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
                if self.apply_bn_1:
                    h1 = batchnorm(h1, g=self.g1, b=self.b1, n=noise_sigma)
                h1 = lrelu(h1)
            h2 = h1
        else:
            # apply first conv layer
            h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
            if self.apply_bn_1:
                h1 = batchnorm(h1, g=self.g1, b=self.b1, n=noise_sigma)
            h1 = lrelu(h1)
            # apply second conv layer (may include downsampling)
            if self.use_pooling:
                h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
                if self.apply_bn_2:
                    h2 = batchnorm(h2, g=self.g2, b=self.b2, n=noise_sigma)
                h2 = lrelu(h2)
                h2 = dnn_pool(h2, (ss,ss), stride=(ss, ss), mode='max', pad=(0, 0))
            else:
                h2 = dnn_conv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
                if self.apply_bn_2:
                    h2 = batchnorm(h2, g=self.g2, b=self.b2, n=noise_sigma)
                h2 = lrelu(h2)
        # apply discriminator layer
        y = dnn_conv(h2, self.wd, subsample=(1, 1), border_mode=(bm, bm))
        y = sigmoid(T.flatten(y, 2)) # flatten to (batch_size, num_preds)
        return h2, y


########################################
# DISCRIMINATOR FULLY CONNECTED MODULE #
########################################

class DiscFCModule(object):
    """
    Module that feeds forward through a single fully connected hidden layer
    and then produces a single scalar "discriminator" output.

    Params:
        fc_dim: dimension of the fully connected layer
        in_dim: dimension of the inputs to the module
        num_layers: 1 or 2, 1 uses no hidden layer and 2 uses a hidden layer
        apply_bn: whether to apply batch normalization at fc layer
        init_func: function for initializing module parameters
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, fc_dim, in_dim, num_layers,
                 apply_bn=True, init_func=None,
                 mod_name='dm_fc'):
        assert ((num_layers == 1) or (num_layers == 2)), \
                "num_layers must be 1 or 2."
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.apply_bn = apply_bn
        self.mod_name = mod_name
        if init_func is None:
            self.init_func = inits.Normal(scale=0.02)
        else:
            self.init_func = init_func
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        if self.num_layers == 2:
            self.w1 = self.init_func((self.in_dim, self.fc_dim),
                                     "{}_w1".format(self.mod_name))
            self.w2 = self.init_func((self.fc_dim, 1),
                                     "{}_w2".format(self.mod_name))
            self.params = [self.w1, self.w2]
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn:
                gain_ifn = inits.Normal(loc=1., scale=0.02)
                bias_ifn = inits.Constant(c=0.)
                self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
                self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
                self.params.extend([self.g1, self.b1])
        else:
            self.w1 = self.init_func((self.in_dim, 1),
                                     "{}_w1".format(self.mod_name))
            self.params = [self.w1]
        return

    def apply(self, input, noise_sigma=None):
        """
        Apply this discriminator module to the given input. This produces a
        scalar discriminator output for each input observation.
        """
        # flatten input to 1d per example
        input = T.flatten(input, 2)
        if self.num_layers == 2:
            # feedforward to fully connected layer
            h1 = T.dot(input, self.w1)
            if self.apply_bn:
                h1 = batchnorm(h1, g=self.g1, b=self.b1, n=noise_sigma)
            h1 = lrelu(h1)
            # feedforward to discriminator outputs
            y = sigmoid(T.dot(h1, self.w2))
        else:
            y = sigmoid(T.dot(input, self.w1))
        return y


#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class GenConvModule(object):
    """
    Module of one "fractionally strided" convolution layer followed by one
    regular convolution layer. Inputs to the fractionally strided convolution
    can optionally be augmented with some random values.

    Params:
        filt_shape: shape for convolution filters -- should be square and odd
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        rand_chans: number of random channels to augment input
        use_rand: flag for whether or not to augment inputs
        num_layers: number of layers in module -- must be 1 or 2
        apply_bn_1: flag for whether to batch normalize following first conv
        apply_bn_2: flag for whether to batch normalize following second conv
        us_stride: upsampling ratio in the fractionally strided convolution
        use_pooling: whether to use unpooling or fractional striding
        init_func: function for initializing module parameters
        mod_name: text name for identifying module in theano graph
        rand_type: whether to use Gaussian or uniform randomness
    """
    def __init__(self, filt_shape, in_chans, out_chans, rand_chans,
                 use_rand=True, num_layers=2,
                 apply_bn_1=True, apply_bn_2=True,
                 us_stride=2, use_pooling=True,
                 init_func=None, mod_name='gm_conv',
                 rand_type='normal'):
        assert ((filt_shape[0] % 2) > 0), \
                "filter dim should be odd (not even)"
        assert ((num_layers == 1) or (num_layers == 2)), \
                "num_layers must be 1 or 2."
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.rand_chans = rand_chans
        self.use_rand = use_rand
        self.num_layers = num_layers
        self.apply_bn_1 = apply_bn_1
        self.apply_bn_2 = apply_bn_2
        self.us_stride = us_stride
        self.use_pooling = use_pooling
        self.mod_name = mod_name
        self.rand_type = rand_type
        self.rng = RandStream(123)
        if init_func is None:
            self.init_func = inits.Normal(scale=0.02)
        else:
            self.init_func = init_func
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        # initialize first layer parameters
        if self.use_rand:
            # random values will be stacked on exogenous input
            self.w1 = self.init_func((self.out_chans, (self.in_chans+self.rand_chans), self.filt_dim, self.filt_dim),
                                     "{}_w1".format(self.mod_name))
        else:
            # random values won't be stacked on exogenous input
            self.w1 = self.init_func((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                         "{}_w1".format(self.mod_name))
        self.params = [self.w1]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            gain_ifn = inits.Normal(loc=1., scale=0.02)
            bias_ifn = inits.Constant(c=0.)
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])

        if self.num_layers == 2:
            # initialize second layer parameters, if required
            self.w2 = self.init_func((self.out_chans, self.out_chans, self.filt_dim, self.filt_dim),
                                     "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
                gain_ifn = inits.Normal(loc=1., scale=0.02)
                bias_ifn = inits.Constant(c=0.)
                self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
                self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
                self.params.extend([self.g2, self.b2])
        return

    def apply(self, input, rand_vals=None):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0]
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
        ss = self.us_stride               # stride for "learned upsampling"
        if self.use_pooling:
            # "unpool" the input if desired
            input = input.repeat(ss, axis=2).repeat(ss, axis=3)
        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        if self.use_rand:
            # augment input with random channels
            if rand_vals is None:
                if self.rand_type == 'normal':
                    rand_vals = self.rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                                dtype=theano.config.floatX)
                else:
                    rand_vals = self.rng.uniform(size=rand_shape, low=-1.0, high=1.0, \
                                                 dtype=theano.config.floatX)
            rand_vals = rand_vals.reshape(rand_shape)
            # stack random values on top of input
            full_input = T.concatenate([rand_vals, input], axis=1)
        else:
            # don't augment input with random channels
            full_input = input
        # apply first convolution, perhaps with fractional striding
        if self.use_pooling:
            h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        else:
            # apply first conv layer (with fractional stride for upsampling)
            h1 = deconv(full_input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
        if self.apply_bn_1:
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
        h1 = relu(h1)
        if self.num_layers == 1:
            # don't apply second conv layer
            h2 = h1
        else:
            # apply second conv layer
            h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
            if self.apply_bn_2:
                h2 = batchnorm(h2, g=self.g2, b=self.b2)
            h2 = relu(h2)
        return h2


####################################
# GENERATOR FULLY CONNECTED MODULE #
####################################

class GenFCModule(object):
    """
    Module that transforms random values through a single fully connected
    layer, and then a linear transform (with another relu, optionally).
    """
    def __init__(self,
                 rand_dim, fc_dim, out_shape,
                 num_layers,
                 apply_bn_1=True, apply_bn_2=True,
                 init_func=None, rand_type='normal',
                 mod_name='dm_fc'):
        assert ((num_layers == 1) or (num_layers == 2)), \
                "num_layers must be 1 or 2."
        assert (len(out_shape) == 3), \
                "out_shape should describe the input to a conv layer."
        self.rand_dim = rand_dim
        self.out_shape = out_shape
        self.out_dim = out_shape[0] * out_shape[1] * out_shape[2]
        self.fc_dim = fc_dim
        self.apply_bn_1 = apply_bn_1
        self.apply_bn_2 = apply_bn_2
        self.num_layers = num_layers
        self.mod_name = mod_name
        self.rand_type = rand_type
        self.rng = RandStream(123)
        if init_func is None:
            self.init_func = inits.Normal(scale=0.02)
        else:
            self.init_func = init_func
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.w1 = self.init_func((self.rand_dim, self.fc_dim),
                                 "{}_w1".format(self.mod_name))
        self.params = [self.w1]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            gain_ifn = inits.Normal(loc=1., scale=0.02)
            bias_ifn = inits.Constant(c=0.)
            self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])
        if self.num_layers == 2:
            self.w2 = self.init_func((self.fc_dim, self.out_dim),
                                     "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
                gain_ifn = inits.Normal(loc=1., scale=0.02)
                bias_ifn = inits.Constant(c=0.)
                self.g2 = gain_ifn((self.out_dim), "{}_g2".format(self.mod_name))
                self.b2 = bias_ifn((self.out_dim), "{}_b2".format(self.mod_name))
                self.params.extend([self.g2, self.b2])
        return

    def apply(self, batch_size=None, rand_vals=None):
        """
        Apply this generator module. Pass _either_ batch_size or rand_vals.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
                "need either batch_size or rand_vals"
        assert ((batch_size is None) or (rand_vals is None)), \
                "need either batch_size or rand_vals"
        if rand_vals is None:
            rand_shape = (batch_size, self.rand_dim)
            if self.rand_type == 'normal':
                rand_vals = self.rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                            dtype=theano.config.floatX)
            else:
                rand_vals = self.rng.uniform(size=rand_shape, low=-1.0, high=1.0, \
                                             dtype=theano.config.floatX)
        else:
            rand_shape = (rand_vals.shape[0], self.rand_dim)
        rand_vals = rand_vals.reshape(rand_shape)
        # always apply first layer
        h1 = T.dot(rand_vals, self.w1)
        if self.apply_bn_1:
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
        h1 = relu(h1)
        if self.num_layers == 1:
            # don't apply second layer
            h2 = h1
        else:
            # do apply second layer
            h2 = T.dot(h1, self.w2)
            if self.apply_bn_2:
                h2 = batchnorm(h2, g=self.g2, b=self.b2)
            h2 = relu(h2)
        # reshape vector outputs for use a conv layer inputs
        h2 = h2.reshape((h2.shape[0], self.out_shape[0], \
                         self.out_shape[1], self.out_shape[2]))
        return h2















##############
# EYE BUFFER #
##############
