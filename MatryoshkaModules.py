import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

#####################################
# BASIC DOUBLE CONVOLUTIONAL MODULE #
#####################################

class BasicConvResModule(object):
    """
    Module with a direct pass-through connection that gets modulated by a pair
    of hidden convolutional layers.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        use_conv: flag for whether to use "internal" convolution layer
        stride: allowed strides are 'double', 'single', and 'half'
        act_func: allowed activations are 'ident', 'relu', and 'lrelu'
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans,
                 use_conv=True, stride='single', act_func='relu',
                 mod_name='basic_conv_res'):
        assert (stride in ['single', 'double', 'half']), \
                "stride must be 'double', 'single', or 'half'."
        assert (act_func in ['ident', 'relu', 'lrelu']), \
                "act_func must be 'ident', 'relu', or 'lrelu'."
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.use_conv = use_conv
        self.stride = stride
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, self.in_chans, 3, 3),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, 3, 3),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w_prj = weight_ifn((self.out_chans, self.in_chans, 3, 3),
                                "{}_w_prj".format(self.mod_name))
        self.g_prj = gain_ifn((self.out_chans), "{}_g_prj".format(self.mod_name))
        self.b_prj = bias_ifn((self.out_chans), "{}_b_prj".format(self.mod_name))
        self.params.extend([self.w_prj, self.g_prj, self.b_prj])
        return

    def apply(self, input):
        """
        Apply this convolutional module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = 1 if (self.stride == 'single') else 2
        if self.use_conv:
            if self.stride in ['double', 'single']:
                # apply first internal conv layer (might downsample)
                h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(1, 1))
                h1 = batchnorm(h1, g=self.g1, b=self.b1)
                h1 = self.act_func(h1)
                # apply second internal conv layer
                h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(1, 1))
                # apply pass-through conv layer (might downsample)
                h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            else:
                # apply first internal conv layer
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(1, 1))
                h1 = batchnorm(h1, g=self.g1, b=self.b1)
                h1 = self.act_func(h1)
                # apply second internal conv layer (might upsample)
                h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(1, 1))
                # apply pass-through conv layer (might upsample)
                h3 = deconv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct pass-through conv layer
            if self.stride in ['double', 'single']:
                h4 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            else:
                h4 = deconv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
        h4 = batchnorm(h4, g=self.g_prj, b=self.b_prj)
        output = self.act_func(h4)
        return output

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
        stride: whether to use 'double', 'single', or 'half' stride.
        apply_bn: whether to apply batch normalization after conv
        act_func: should be "ident", "relu", or "lrelu"
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans,
                 stride='single', apply_bn=True, act_func='ident',
                 mod_name='basic_conv'):
        assert ((filt_shape[0] % 2) > 0), "filter dim should be odd (not even)"
        assert (stride in ['single', 'double', 'half']), \
                "stride should be 'single', 'double', or 'half'."
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = stride
        self.apply_bn = apply_bn
        self.act_func = act_func
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.params = [self.w1]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn:
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])
        return

    def apply(self, input, rand_vals=None, rand_shapes=False, noise_sigma=None):
        """
        Apply this convolutional module to the given input.
        """
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
        # apply first conv layer
        if self.stride == 'single':
            # normal, 1x1 stride
            h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        elif self.stride == 'double':
            # downsampling, 2x2 stride
            h1 = dnn_conv(input, self.w1, subsample=(2, 2), border_mode=(bm, bm))
        else:
            # upsampling, 0.5x0.5 stride
            h1 = deconv(input, self.w1, subsample=(2, 2), border_mode=(bm, bm))
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
        if rand_shapes:
            result = [h1, input.shape]
        else:
            result = h1
        return result

#############################################
# DISCRIMINATOR DOUBLE CONVOLUTIONAL MODULE #
#############################################

class DiscConvResModule(object):
    """
    Module of one regular convolution layer followed by one "fractionally
    strided convolution layer. Has a direct pass-through connection.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        use_conv: flag for whether to use "internal" convolution layer
        ds_stride: downsampling ratio in the fractionally strided convolution
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans,
                 use_conv=True, ds_stride=2,
                 mod_name='dm_conv'):
        assert ((ds_stride == 1) or (ds_stride == 2)), \
                "ds_stride must be 1 or 2."
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.use_conv = use_conv
        self.ds_stride = ds_stride
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, self.in_chans, 3, 3),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, 3, 3),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w_prj = weight_ifn((self.out_chans, self.in_chans, 3, 3),
                                "{}_w_prj".format(self.mod_name))
        self.g_prj = gain_ifn((self.out_chans), "{}_g_prj".format(self.mod_name))
        self.b_prj = bias_ifn((self.out_chans), "{}_b_prj".format(self.mod_name))
        self.params.extend([self.w_prj, self.g_prj, self.b_prj])
        # initialize weights for the "discrimination" layer
        self.wd = weight_ifn((1, self.out_chans, 3, 3),
                             "{}_wd".format(self.mod_name))
        return

    def apply(self, input, noise_sigma=None):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.ds_stride         # stride for "learned downsampling"
        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(1, 1))
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
            h1 = lrelu(h1)
            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(1, 1))
            # apply direct input->output "projection" layer
            h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))

            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
            h4 = batchnorm(h4, g=self.g_prj, b=self.b_prj)
            output = lrelu(h4)
        else:
            # apply direct input->output "projection" layer
            h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            h3 = batchnorm(h3, g=self.g_prj, b=self.b_prj)
            output = lrelu(h3)

        # apply discriminator layer
        y = dnn_conv(output, self.wd, subsample=(1, 1), border_mode=(1, 1))
        y = T.flatten(y, 2)
        return [output, y]

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
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans, num_layers=2,
                 apply_bn_1=True, apply_bn_2=True, ds_stride=2,
                 use_pooling=True, mod_name='dm_conv'):
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize params for first layer and discriminator layer
        self.w1 = weight_ifn((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.wd = weight_ifn((1, self.out_chans, self.filt_dim, self.filt_dim),
                             "{}_wd".format(self.mod_name))
        self.params = [self.w1, self.wd]
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])

        if self.num_layers == 2:
            # initialize parameters for second layer
            self.w2 = weight_ifn((self.out_chans, self.out_chans, self.filt_dim, self.filt_dim),
                                 "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
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
        y = T.flatten(y, 2)
        return [h2, y]


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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        if self.num_layers == 2:
            self.w1 = weight_ifn((self.in_dim, self.fc_dim),
                                 "{}_w1".format(self.mod_name))
            self.w2 = weight_ifn((self.fc_dim, 1),
                                 "{}_w2".format(self.mod_name))
            self.params = [self.w1, self.w2]
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn:
                self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
                self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
                self.params.extend([self.g1, self.b1])
        else:
            self.w1 = weight_ifn((self.in_dim, 1),
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
            h2 = T.dot(h1, self.w2)
            y = h2
        else:
            h2 = T.dot(input, self.w1)
            y = h2
        return [h2, y]



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
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans, rand_chans,
                 use_rand=True, num_layers=2,
                 apply_bn_1=True, apply_bn_2=True,
                 us_stride=2, use_pooling=True,
                 mod_name='gm_conv'):
        assert ((filt_shape[0] % 2) > 0), \
                "filter dim should be odd (not even)"
        assert ((num_layers == 1) or (num_layers == 2)), \
                "num_layers must be 1 or 2."
        assert ((us_stride == 1) or (us_stride == 2)), \
                "us_stride must be 1 or 2."
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first layer parameters
        self.w1 = weight_ifn((self.out_chans, (self.in_chans+self.rand_chans), self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.params.extend([self.w1])
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])

        if self.num_layers == 2:
            # initialize second layer parameters, if required
            self.w2 = weight_ifn((self.out_chans, self.out_chans, self.filt_dim, self.filt_dim),
                                 "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
                self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
                self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
                self.params.extend([self.g2, self.b2])
        return

    def apply(self, input, rand_vals=None, rand_shapes=False):
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
        # augment input with random channels
        if rand_vals is None:
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                      dtype=theano.config.floatX)
        if not self.use_rand:
            rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape # return vals must be theano vars
        # stack random values on top of input
        full_input = T.concatenate([rand_vals, input], axis=1)

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
        if rand_shapes:
            result = [h2, rand_shape]
        else:
            result = h2
        return result


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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        #
        self.w1 = weight_ifn((self.rand_dim, self.fc_dim),
                             "{}_w1".format(self.mod_name))
        self.params.extend([self.w1])
        # make gains and biases for transforms that will get batch normed
        if self.apply_bn_1:
            self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
            self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
            self.params.extend([self.g1, self.b1])
        if self.num_layers == 2:
            self.w2 = weight_ifn((self.fc_dim, self.out_dim),
                                 "{}_w2".format(self.mod_name))
            self.params.extend([self.w2])
            # make gains and biases for transforms that will get batch normed
            if self.apply_bn_2:
                self.g2 = gain_ifn((self.out_dim), "{}_g2".format(self.mod_name))
                self.b2 = bias_ifn((self.out_dim), "{}_b2".format(self.mod_name))
                self.params.extend([self.g2, self.b2])
        return

    def apply(self, batch_size=None, rand_vals=None, rand_shapes=False):
        """
        Apply this generator module. Pass _either_ batch_size or rand_vals.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
                "need either batch_size or rand_vals"
        assert ((batch_size is None) or (rand_vals is None)), \
                "need either batch_size or rand_vals"
        if rand_vals is None:
            rand_shape = (batch_size, self.rand_dim)
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                        dtype=theano.config.floatX)
        else:
            rand_shape = (rand_vals.shape[0], self.rand_dim)
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape
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
        # reshape vector outputs for use as conv layer inputs
        h2 = h2.reshape((h2.shape[0], self.out_shape[0], \
                         self.out_shape[1], self.out_shape[2]))
        if rand_shapes:
            result = [h2, rand_shape]
        else:
            result = h2
        return result


##################################################
# GENERATOR DOUBLE CONVOLUTIONAL RESIDUAL MODULE #
##################################################

class GenConvDblResModule(object):
    """
    Module structure is based on the "bottleneck" modules of MSR's residual
    network that won the 2015 Imagenet competition(s).

    First layer is a 1x1 convolutional layer with batch normalization.

    Second layer is a 3x3 convolutional layer with batch normalization.

    Third layer is a 1x1 convolutional layer.

    Output of the third layer is added to the result of a conv layer applied
    directly to the input, and the resulting sum is batch normalized.

    Process: IN -> fc1(IN) -> conv(fc1) -> fc2(conv) -> relu(fc2 + IN)

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in convolutional layer
        rand_chans: number of random channels to augment input
        use_rand: flag for whether or not to augment inputs with noise
        use_conv: flag for whether or not to use internal conv layers
        us_stride: upsampling ratio in the fractionally strided convolution
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, rand_chans,
                 use_rand=True, use_conv=True, us_stride=2,
                 mod_name='gm_conv'):
        assert ((us_stride == 1) or (us_stride == 2)), \
            "upsampling stride (i.e. us_stride) must be 1 or 2."
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.rand_chans = rand_chans
        self.use_rand = use_rand
        self.use_conv = use_conv
        self.us_stride = us_stride
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        #
        # Initialize parameters for first 1x1 convolutional layer.
        #
        # input shape : (batch, in_chans+rand_chans, rows, cols)
        # output shape: (batch, conv_chans, rows, cols)
        #
        self.w_fc1 = weight_ifn((self.conv_chans, (self.in_chans+self.rand_chans), 1, 1),
                                "{}_w_fc1".format(self.mod_name))
        self.g_fc1 = gain_ifn((self.conv_chans), "{}_g_fc1".format(self.mod_name))
        self.b_fc1 = bias_ifn((self.conv_chans), "{}_b_fc1".format(self.mod_name))
        self.params.extend([self.w_fc1, self.g_fc1, self.b_fc1])
        #
        # Initialize parameters for 3x3 convolutional layer.
        #
        # input shape : (batch, conv_chans, rows, cols)
        # output shape: (batch, conv_chans, rows/us_stride, cols/us_stride)
        #
        self.w_conv = weight_ifn((self.conv_chans, self.conv_chans, 3, 3),
                                 "{}_w_conv".format(self.mod_name))
        self.g_conv = gain_ifn((self.conv_chans), "{}_g_conv".format(self.mod_name))
        self.b_conv = bias_ifn((self.conv_chans), "{}_b_conv".format(self.mod_name))
        self.params.extend([self.w_conv, self.g_conv, self.b_conv])
        #
        # Initialize parameters for second 1x1 convolutional layer.
        #
        # input shape : (batch, conv_chans, rows/us_stride, cols/us_stride)
        # output shape: (batch, out_chans, rows/us_stride, cols/us_stride)
        #
        self.w_fc2 = weight_ifn((self.out_chans, self.conv_chans, 1, 1),
                                "{}_w_fc2".format(self.mod_name))
        self.g_fc2 = gain_ifn((self.out_chans), "{}_g_fc2".format(self.mod_name))
        self.b_fc2 = bias_ifn((self.out_chans), "{}_b_fc2".format(self.mod_name))
        self.params.extend([self.w_fc2, self.g_fc2, self.b_fc2])
        #
        # Initialize parameters for pass-through convolution and merging layer.
        #
        # input shape : (batch, in_chans, rows, cols)
        # output shape: (batch, out_chans, rows, cols)
        #
        self.w_out = weight_ifn((self.out_chans, (self.in_chans+self.rand_chans), 3, 3),
                                "{}_w_out".format(self.mod_name))
        self.g_out = gain_ifn((self.out_chans), "{}_g_out".format(self.mod_name))
        self.b_out = bias_ifn((self.out_chans), "{}_b_out".format(self.mod_name))
        self.params.extend([self.w_out, self.g_out, self.b_out])
        return

    def apply(self, input, rand_vals=None, rand_shapes=False):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.us_stride         # stride for upsampling

        # get shape for random values that we'll append to module input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        if rand_vals is None:
            # generate random values to append to module input
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0,
                                      dtype=theano.config.floatX)
        if not self.use_rand:
            # mask out random values, so they won't get used
            rand_vals = 0.0 * rand_vals
        rand_shape = rand_vals.shape # return vals must be theano vars

        # stack random values on top of input
        input_and_rvs = T.concatenate([rand_vals, input], axis=1)

        if self.use_conv:
            # apply first 1x1 conv layer
            h1 = dnn_conv(input_and_rvs, self.w_fc1, subsample=(1, 1), border_mode=(0, 0))
            h1 = batchnorm(h1, g=self.g_fc1, b=self.b_fc1)
            h1 = relu(h1)
            # apply 3x3 conv layer (might upsample)
            h2 = deconv(h1, self.w_conv, subsample=(ss, ss), border_mode=(1, 1))
            h2 = batchnorm(h2, g=self.g_conv, b=self.b_conv)
            h2 = relu(h2)
            # apply second 1x1 conv layer
            h3 = dnn_conv(h2, self.w_fc2, subsample=(1, 1), border_mode=(0, 0))
            # apply pass-through convolution to input (might upsample)
            h4 = deconv(input_and_rvs, self.w_out, subsample=(ss, ss), border_mode=(1, 1))
            # merge output of internal conv layers with the pass-through result
            output = h3 + h4
        else:
            # apply pass-through convolution to input (might upsample)
            output = deconv(input_and_rvs, self.w_out, subsample=(ss, ss), border_mode=(1, 1))
        output = batchnorm(output, g=self.g_out, b=self.b_out)
        output = relu(output)
        # decide what to return: output only, or output and rand_vals.shape...
        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class GenConvResModule(object):
    """
    Module of one "fractionally strided" convolution layer followed by one
    regular convolution layer. Inputs to the fractionally strided convolution
    can optionally be augmented with some random values.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        rand_chans: number of random channels to augment input
        use_rand: flag for whether or not to augment inputs
        use_conv: flag for whether to use "internal" convolution layer
        us_stride: upsampling ratio in the fractionally strided convolution
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, rand_chans,
                 use_rand=True, use_conv=True, us_stride=2,
                 mod_name='gm_conv'):
        assert ((us_stride == 1) or (us_stride == 2)), \
                "us_stride must be 1 or 2."
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.rand_chans = rand_chans
        self.use_rand = use_rand
        self.use_conv = use_conv
        self.us_stride = us_stride
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.in_chans+self.rand_chans), 3, 3),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, 3, 3),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w_prj = weight_ifn((self.out_chans, (self.in_chans+self.rand_chans), 3, 3),
                                "{}_w_prj".format(self.mod_name))
        self.g_prj = gain_ifn((self.out_chans), "{}_g_prj".format(self.mod_name))
        self.b_prj = bias_ifn((self.out_chans), "{}_b_prj".format(self.mod_name))
        self.params.extend([self.w_prj, self.g_prj, self.b_prj])
        return

    def apply(self, input, rand_vals=None, rand_shapes=False):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.us_stride         # stride for "learned upsampling"

        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        # augment input with random channels
        if rand_vals is None:
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                      dtype=theano.config.floatX)
        if not self.use_rand:
            rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape # return vals must be theano vars

        # stack random values on top of input
        full_input = T.concatenate([rand_vals, input], axis=1)

        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(1, 1))
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
            h1 = relu(h1)
            # apply second internal conv layer
            h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(1, 1))
            # apply direct input->output "projection" layer
            h3 = deconv(full_input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct input->output "projection" layer
            h4 = deconv(full_input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
        h4 = batchnorm(h4, g=self.g_prj, b=self.b_prj)
        output = relu(h4)

        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result



#
# TODO: Implement "InfConvMergeModule" and "InfFCModule".
#
# InfConvMergeModule: this module merges bottom-up and top-down information, in
#                     order to place an approximate posterior over a 2d grid of
#                     vectors of Gaussian latent variables.
#
# InfFCModule: this module sits on top of a bottom-up inference network, in
#              in order to place an approximate posterior over a vector of
#              Gaussian latent variables, at the initial layer of a deep,
#              directed generative model.
#

#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class InfConvMergeModule(object):
    """
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        conv_chans: number of channels in the "internal" convolution layer
        use_conv: flag for whether to use "internal" convolution layer
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, rand_chans, conv_chans,
                 use_conv=True,
                 mod_name='gm_conv'):
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
        self.use_conv = use_conv
        self.us_stride = us_stride
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this generator module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.td_chans+self.bu_chans), 3, 3),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((2*self.rand_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((2*self.rand_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w_out = weight_ifn((2*self.rand_chans, (self.td_chans+self.bu_chans), 3, 3),
                                "{}_w_out".format(self.mod_name))
        self.g_out = gain_ifn((2*self.rand_chans), "{}_g_out".format(self.mod_name))
        self.b_out = bias_ifn((2*self.rand_chans), "{}_b_out".format(self.mod_name))
        self.params.extend([self.w_out, self.g_out, self.b_out])
        return

    def apply(self, input, rand_vals=None, rand_shapes=False):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.us_stride         # stride for "learned upsampling"

        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        # augment input with random channels
        if rand_vals is None:
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                      dtype=theano.config.floatX)
        if not self.use_rand:
            rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape # return vals must be theano vars

        # stack random values on top of input
        full_input = T.concatenate([rand_vals, input], axis=1)

        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(1, 1))
            h1 = batchnorm(h1, g=self.g1, b=self.b1)
            h1 = relu(h1)
            # apply second internal conv layer
            h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(1, 1))
            # apply direct input->output "projection" layer
            h3 = deconv(full_input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct input->output "projection" layer
            h4 = deconv(full_input, self.w_prj, subsample=(ss, ss), border_mode=(1, 1))
        h4 = batchnorm(h4, g=self.g_prj, b=self.b_prj)
        output = relu(h4)

        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


##############
# EYE BUFFER #
##############
