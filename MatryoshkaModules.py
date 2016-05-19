import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import inits
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, \
                    add_noise, log_mean_exp

from lib.theano_utils import floatX, sharedX

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify(leak=0.1)
bce = T.nnet.binary_crossentropy
tanh = activations.Tanh()
elu = activations.ELU()


def tanh_clip(x, scale=10.0):
    """
    Do soft "tanh" clipping to put data in range -scale....+scale.
    """
    x = scale * tanh((1.0 / scale) * x)
    return x


def fc_drop_func(x, unif_drop, share_mask=False):
    """
    Helper func for applying uniform dropout.
    """
    # dumb dropout, no rescale (assume MC dropout usage)
    if not share_mask:
        # make separate drop mask for each input
        if unif_drop > 0.01:
            r = cu_rng.uniform(x.shape, dtype=theano.config.floatX)
            x = x * (r > unif_drop)
    else:
        # use the same mask for entire batch
        if unif_drop > 0.01:
            r = cu_rng.uniform((x.shape[1],), dtype=theano.config.floatX)
            r = r.dimshuffle('x', 0)
            x = x * (r > unif_drop)
    return x


def conv_drop_func(x, unif_drop, chan_drop, share_mask=False):
    """
    Helper func for applying uniform and/or channel-wise dropout.
    """
    # dumb dropout, no rescale (assume MC dropout usage)
    if not share_mask:
        # make a separate drop mask for each input
        if unif_drop > 0.01:
            ru = cu_rng.uniform(x.shape, dtype=theano.config.floatX)
            x = x * (ru > unif_drop)
        if chan_drop > 0.01:
            rc = cu_rng.uniform((x.shape[0], x.shape[1]),
                                dtype=theano.config.floatX)
            chan_mask = (rc > chan_drop)
            x = x * chan_mask.dimshuffle(0, 1, 'x', 'x')
    else:
        # use the same mask for entire batch
        if unif_drop > 0.01:
            ru = cu_rng.uniform((x.shape[1], x.shape[2], x.shape[3]),
                                dtype=theano.config.floatX)
            ru = ru.dimshuffle('x', 0, 1, 2)
            x = x * (ru > unif_drop)
        if chan_drop > 0.01:
            rc = cu_rng.uniform((x.shape[1],),
                                dtype=theano.config.floatX)
            chan_mask = (rc > chan_drop)
            x = x * chan_mask.dimshuffle('x', 0, 'x', 'x')
    return x


def switchy_bn(acts, g=None, b=None, use_gb=True, n=None):
    """
    Helper function for optionally applying secondary shift+scale in BN.
    """
    if use_gb and (not (g is None) or (b is None)):
        bn_acts = batchnorm(acts, g=g, b=b, n=n)
    else:
        bn_acts = batchnorm(acts, n=n)
    return bn_acts


class BasicFCPertModule(object):
    """
    Module with a linear short-cut connection that gets combined with the
    output of a linear->activation->linear module.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        fc_chans: number of channels in the "internal" layer
        use_fc: flag for whether to use "internal" layer
        act_func: --
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to apply batch normalization
        use_bn_params: whether to use post-processing params for BN
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, fc_chans,
                 use_fc=True, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True,
                 mod_name='basic_fc_res'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.fc_chans = fc_chans
        self.use_fc = use_fc
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.mod_name = mod_name
        self.use_bn_params = use_bn_params
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.in_chans, self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.fc_chans, self.out_chans),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w3 = weight_ifn((self.in_chans, self.out_chans),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this fully-connected module to some input.
        """
        # apply uniform and/or channel-wise dropout if desired
        input = fc_drop_func(input, self.unif_drop, share_mask=share_mask)
        if self.use_fc:
            # apply first internal linear layer and activation
            h1 = T.dot(input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # apply second internal linear layer
            h2 = T.dot(h1, self.w2)
            if self.apply_bn:
                h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h2 = h2 + self.b2.dimshuffle('x', 0)
                h2 = add_noise(h2, noise=noise)
            # apply short-cut linear layer
            # h3 = T.dot(input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + input  # + h3
        else:
            # apply short-cut linear layer
            h4 = T.dot(input, self.w3)
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b3.dimshuffle('x', 0)
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
        if rand_shapes:
            result = [output, input.shape]
        else:
            result = output
        return result


####################################
# BASIC FULLY-CONNECTED GRU MODULE #
####################################

class FancyFCGRUModule(object):
    """
    GRU-type module that optionally takes an exogenous input.

    Params:
        state_chans: dimension of GRU state
        rand_chans: dimension of stochastic inputs (0 or None means no input)
        act_func: --
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to apply batch normalization
        use_bn_params: whether to use post-processing params for BN
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 state_chans, rand_chans,
                 act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True,
                 mod_name='basic_fc_gru'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        if ((rand_chans is None) or (rand_chans == 0)):
            self.rand_chans = 0
        else:
            self.rand_chans = rand_chans
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.mod_name = mod_name
        self.use_bn_params = use_bn_params
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize gating parameters
        self.w1 = weight_ifn((self.state_chans + self.rand_chans, 2 * self.state_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((2 * self.state_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize state update parameters
        self.w2 = weight_ifn((self.state_chans + self.rand_chans, self.state_chans),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.state_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.state_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        return

    def share_params(self, source_module):
        """
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        """
        self.params = []
        # share gating layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # share update layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this fully-connected module to some input.
        """
        # if rand_vals was given, add it to the gating input
        if self.rand_chans > 0:
            g_in = T.concatenate([input, rand_vals], axis=1)
        else:
            g_in = input
            rand_vals = T.alloc(0.0, input.shape[0], 4)
        rand_shape = rand_vals.shape
        # compute gate stuff
        h1 = T.dot(g_in, self.w1)
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x', 0)
            h1 = add_noise(h1, noise=noise)
        h1 = sigmoid(h1 + 1.)
        # split information for update/recall gates
        r = h1[:, :self.state_chans]
        z = h1[:, self.state_chans:]

        # apply recall gate to input and compute state update proposal
        if self.rand_chans > 0:
            u_in = T.concatenate([r * input, rand_vals], axis=1)
        else:
            u_in = r * input
        h2 = T.dot(u_in, self.w2)
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2.dimshuffle('x', 0)
            h2 = add_noise(h2, noise=noise)
        update = self.act_func(h2)

        # compute the updated GRU state
        output = (z * input) + ((1. - z) * update)

        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


###############################
# BASIC FULLY-CONNECTED LAYER #
###############################

class BasicFCModule(object):
    """
    Simple fully-connected layer for use anywhere?

    Params:
        in_chans: number of channels in input
        out_chans: number of channels to produce as output
        apply_bn: whether to apply batch normalization before activation
        act_func: --
        unif_drop: drop rate for uniform dropout
        use_bn_params: whether to use params for BN
        use_noise: whether to use the provided noise during apply()
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, in_chans, out_chans,
                 apply_bn=True,
                 act_func='ident',
                 unif_drop=0.0,
                 use_bn_params=True,
                 use_noise=True,
                 mod_name='basic_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.apply_bn = apply_bn
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.use_bn_params = use_bn_params
        self.use_noise = use_noise
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.in_chans, self.out_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this convolutional module to the given input.
        """
        noise = noise if self.use_noise else None
        # apply uniform and/or channel-wise dropout if desired
        input = fc_drop_func(input, self.unif_drop, share_mask=share_mask)
        # linear transform followed by activations and stuff
        h1 = T.dot(input, self.w1)
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x', 0)
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        if rand_shapes:
            result = [h1, input.shape]
        else:
            result = h1
        return result


###########################################
# BASIC CONVOLUTIONAL PERTURBATION MODULE #
###########################################

class BasicConvPertModule(object):
    """
    Module whose output is computed by adding its input to a non-linear
    function of its input, and then applying a non-linearity.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        filt_shape: size of filters (either (3, 3) or (5, 5))
        use_conv: flag for whether to use "internal" convolution layer
        stride: allowed strides are 'double', 'single', and 'half'
        act_func: --
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_bn_params: whether to use post-processing params for BN
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, filt_shape,
                 use_conv=True, stride='single', act_func='relu',
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, mod_name='basic_conv_res'):
        assert (stride in ['single']), \
            "stride must be 'single'."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3,3) or (5,5)."
        assert (in_chans == out_chans), \
            "in_chans and out_chans must be the same."
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.filt_dim = filt_shape[0]
        self.use_conv = use_conv
        self.stride = stride
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.mod_name = mod_name
        self.use_bn_params = use_bn_params
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, self.in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize alternate conv layer parameters
        self.w3 = weight_ifn((self.out_chans, self.in_chans, fd, fd),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        # initialize first conv layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize alternate conv layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this convolutional module to some input.
        """
        bm = (self.filt_dim - 1) // 2
        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)

        # apply_conv(x, w, g, b, noise, use_gb, apply_bn, stride, border_mode)

        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
            if self.apply_bn:
                h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h2 = h2 + self.b2.dimshuffle('x', 0, 'x', 'x')
                h2 = add_noise(h2, noise=noise)
            # combine non-linear and linear transforms of input...
            h3 = input + h2
        else:
            # apply standard conv layer
            h3 = dnn_conv(input, self.w3, subsample=(1, 1), border_mode=(bm, bm))
        # post-process the perturbed input
        if self.apply_bn:
            h3 = switchy_bn(h3, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h3 = h3 + self.b3.dimshuffle('x', 0, 'x', 'x')
            h3 = add_noise(h3, noise=noise)
        output = self.act_func(h3)
        if rand_shapes:
            result = [output, input.shape]
        else:
            result = output
        return result


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
        act_func: --
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        use_bn_params: whether to use params for BN
        use_noise: whether to use provided noise during apply()
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, filt_shape, in_chans, out_chans,
                 stride='single', apply_bn=True, act_func='ident',
                 unif_drop=0.0, chan_drop=0.0,
                 use_bn_params=True, rescale_output=False,
                 use_noise=True,
                 mod_name='basic_conv'):
        if not (filt_shape[0] == 2):
            assert ((filt_shape[0] % 2) > 0), "filter dim should be odd (not even)"
        assert (stride in ['single', 'double', 'half']), \
            "stride should be 'single', 'double', or 'half'."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.stride = stride
        self.apply_bn = apply_bn
        self.rescale_output = rescale_output
        self.act_func = act_func
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.use_bn_params = use_bn_params
        self.use_noise = use_noise
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params = [self.w1, self.g1, self.b1]
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this convolutional module to the given input.
        """
        noise = noise if self.use_noise else None
        if (self.filt_dim == 2):
            # when self.filt_dim == 2, we should be doing downsampling, and
            # when we do this, we set the padding width to 0.
            bm = 0

        else:
            bm = int((self.filt_dim - 1) / 2)

        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)
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
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            if self.rescale_output:
                h1 = h1 * self.g1.dimshuffle('x', 0, 'x', 'x')
            h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        if rand_shapes:
            result = [h1, input.shape]
        else:
            result = h1
        return result


#########################################
# DOUBLE GENERATOR CONVOLUTIONAL MODULE #
#########################################

class BasicConvGRUModule(object):
    """
    Test module.
    """
    def __init__(self, filt_shape, in_chans, out_chans,
                 act_func='tanh', unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True, use_bn_params=True, stride='single',
                 mod_name='gm_conv'):
        assert ((in_chans == out_chans)), \
            "in_chans == out_chans is required."
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3,3) or (5,5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.out_chans = out_chans
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_bn_params = use_bn_params
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize gate layer parameters
        self.w1 = weight_ifn((2 * self.in_chans, self.in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((2 * self.in_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.in_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize first new state layer parameters
        self.w2 = weight_ifn((self.in_chans, self.in_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.in_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.in_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        return

    def share_params(self, source_module):
        """
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        """
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False,
              noise=None, share_mask=False):
        """
        Apply this generator module to some input.
        """
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        # compute update gate and remember gate
        h = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h = switchy_bn(h, g=self.g1, b=self.b1, n=noise,
                           use_gb=self.use_bn_params)
        else:
            h = h + self.b1.dimshuffle('x', 0, 'x', 'x')
            h = add_noise(h, noise=noise)
        h = sigmoid(h + 1.)
        u = h[:, :self.in_chans, :, :]
        r = h[:, self.in_chans:, :, :]
        # compute new state proposal -- include hidden layer
        s = dnn_conv((r * input), self.w2, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            s = switchy_bn(s, g=self.g2, b=self.b2, n=noise,
                           use_gb=self.use_bn_params)
        else:
            s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
            s = add_noise(s, noise=noise)
        s = self.act_func(s)
        # combine initial state and proposed new state based on u
        output = (u * input) + ((1. - u) * s)
        #
        if rand_shapes:
            result = [output, input.shape]
        else:
            result = output
        return result


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
        use_fc: whether or not to use the hidden layer
        apply_bn: whether to apply batch normalization at fc layer
        unif_drop: drop rate for uniform dropout
        act_func: --
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, fc_dim, in_dim, use_fc,
                 apply_bn=True,
                 unif_drop=0.0,
                 act_func='lrelu',
                 use_bn_params=True,
                 mod_name='dm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.use_fc = use_fc
        self.apply_bn = apply_bn
        self.unif_drop = unif_drop
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.use_bn_params = use_bn_params
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.in_dim, self.fc_dim),
                             "{}_w1".format(self.mod_name))
        self.w2 = weight_ifn((self.fc_dim, 1),
                             "{}_w2".format(self.mod_name))
        self.w3 = weight_ifn((self.in_dim, 1),
                             "{}_w3".format(self.mod_name))
        self.params = [self.w1, self.w2, self.w3]
        # make gains and biases for transforms that will get batch normed
        self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
        self.params.extend([self.g1, self.b1])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.w3.set_value(floatX(param_dict['w3']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        return param_dict

    def apply(self, input, noise=None, share_mask=False):
        """
        Apply this discriminator module to the given input. This produces a
        scalar discriminator output for each input observation.
        """
        # flatten input to 1d per example
        input = T.flatten(input, 2)
        input = fc_drop_func(input, self.unif_drop, share_mask=share_mask)
        if self.use_fc:
            # feedforward to fully connected layer
            h1 = T.dot(input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # compute discriminator output from fc layer and input
            h2 = T.dot(h1, self.w2) + T.dot(input, self.w3)
        else:
            h2 = T.dot(input, self.w3)
        return h2


#############################################
# DISCRIMINATOR DOUBLE CONVOLUTIONAL MODULE #
#############################################

class DiscConvResModule(object):
    """
    Module of one regular convolution layer followed by one "fractionally
    strided convolution layer. Includes a direct short-cut connection.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        filt_shape: size of filters (either (3, 3) or (5, 5))
        use_conv: flag for whether to use "internal" convolution layer
        ds_stride: downsampling ratio in the fractionally strided convolution
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        act_func: ---
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, filt_shape,
                 use_conv=True, ds_stride=2,
                 unif_drop=0.0, chan_drop=0.0,
                 act_func='lrelu', apply_bn=True,
                 use_bn_params=True,
                 mod_name='dm_conv'):
        assert ((ds_stride == 1) or (ds_stride == 2)), \
            "ds_stride must be 1 or 2."
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3,3) or (5,5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.filt_dim = filt_shape[0]
        self.use_conv = use_conv
        self.ds_stride = ds_stride
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.apply_bn = apply_bn
        self.use_bn_params = use_bn_params
        self.mod_name = mod_name
        self._init_params()

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, self.in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w3 = weight_ifn((self.out_chans, self.in_chans, fd, fd),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        # initialize weights for the "discrimination" layer
        self.wd = weight_ifn((1, self.out_chans, 3, 3),
                             "{}_wd".format(self.mod_name))
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        self.wd.set_value(floatX(param_dict['wd']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        param_dict['wd'] = self.wd.get_value(borrow=False)
        return param_dict

    def apply(self, input, noise=None, share_mask=False):
        """
        Apply this convolutional discriminator module to some input.
        """
        ss = self.ds_stride            # stride for "learned downsampling"
        bm = (self.filt_dim - 1) // 2  # set border mode for the convolutions
        # apply dropout to input
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)
        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
                h1 = add_noise(h1, noise=noise)
            # apply activation and maybe dropout
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
            # apply direct input->output short-cut layer
            h3 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
            if self.apply_bn:
                h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h4 = h4 + self.b3.dimshuffle('x', 0, 'x', 'x')
                h4 = add_noise(h4, noise=noise)
            output = self.act_func(h4)
        else:
            # apply direct input->output short-cut layer
            h3 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            if self.apply_bn:
                h3 = switchy_bn(h3, g=self.g3, b=self.b3, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h3 = h3 + self.b3.dimshuffle('x', 0, 'x', 'x')
                h3 = add_noise(h3, noise=noise)
            output = self.act_func(h3)

        # apply discriminator layer
        d_in = conv_drop_func(output, self.unif_drop, self.chan_drop,
                              share_mask=share_mask)
        y = dnn_conv(d_in, self.wd, subsample=(1, 1), border_mode=(1, 1))
        y = T.flatten(y, 2)
        return [output, y]


##################################################
# FULLY CONNECTED MODULE -- FOR TOP OF GENERATOR #
##################################################

class GenTopModule(object):
    """
    Module that transforms random values through a single fully-connected
    layer, and adds this to a linear transform.
    """
    def __init__(self,
                 rand_dim, fc_dim, out_shape,
                 rand_shape=None,
                 use_fc=True, use_sc=False, apply_bn=True,
                 unif_drop=0.0,
                 act_func='relu',
                 use_bn_params=True,
                 aux_dim=None,
                 mod_name='dm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.rand_dim = rand_dim
        self.out_shape = out_shape
        if len(self.out_shape) == 1:
            # output goes to FC layer
            self.out_dim = out_shape[0]
        else:
            # output goes to conv layer
            self.out_dim = out_shape[0] * out_shape[1] * out_shape[2]
        self.rand_shape = rand_shape
        self.fc_dim = fc_dim
        self.use_fc = use_fc
        self.use_sc = use_sc
        self.apply_bn = apply_bn
        self.unif_drop = unif_drop
        self.use_bn_params = use_bn_params
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.aux_dim = aux_dim
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first layer parameters
        self.w1 = weight_ifn((self.rand_dim, self.fc_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second layer parameters
        self.w2 = weight_ifn((self.fc_dim, self.out_dim),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_dim), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_dim), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize single layer parameters
        self.w3 = weight_ifn((self.rand_dim, self.out_dim),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_dim), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_dim), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        # make params for the auxiliary task if desired
        if self.aux_dim is not None:
            self.wa = weight_ifn((self.fc_dim, self.aux_dim),
                                 "{}_wa".format(self.mod_name))
            self.ba = bias_ifn((self.aux_dim), "{}_ba".format(self.mod_name))
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        if self.aux_dim is not None:
            self.wa.set_value(floatX(param_dict['wa']))
            self.ba.set_value(floatX(param_dict['ba']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        if self.aux_dim is not None:
            param_dict['wa'] = self.wa.get_value(borrow=False)
            param_dict['ba'] = self.ba.get_value(borrow=False)
        return param_dict

    def apply(self, batch_size=None, rand_vals=None, rand_shapes=False,
              share_mask=False, noise=None):
        """
        Apply this generator module. Pass _either_ batch_size or rand_vals.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
            "need either batch_size or rand_vals"
        assert ((batch_size is None) or (rand_vals is None)), \
            "need either batch_size or rand_vals"

        if rand_vals is None:
            # we need to generate some latent variables
            rand_shape = (batch_size, self.rand_dim)
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0,
                                      dtype=theano.config.floatX)
        else:
            # get the shape of the incoming latent variables
            rand_shape = (rand_vals.shape[0], self.rand_dim)
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape

        if self.use_fc:
            h1 = T.dot(rand_vals, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            if self.use_sc:
                h2 = T.dot(h1, self.w2) + T.dot(rand_vals, self.w3)
            else:
                h2 = T.dot(h1, self.w2)
        else:
            h2 = T.dot(rand_vals, self.w3)
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b3.dimshuffle('x', 0)
            h2 = add_noise(h2, noise=noise)
        h2 = self.act_func(h2)
        if len(self.out_shape) > 1:
            # reshape vector outputs for use as conv layer inputs
            h2 = h2.reshape((h2.shape[0], self.out_shape[0],
                            self.out_shape[1], self.out_shape[2]))
        if rand_shapes:
            result = [h2, rand_shape]
        else:
            result = h2

        if self.aux_dim is not None:
            # compute auxiliary output as a linear function of the  hidden
            # layer's activations.
            h_aux = T.dot(h1, self.wa) + self.ba.dimshuffle('x', 0)
            result = [h2, h_aux]
        return result


#########################################
# DOUBLE GENERATOR CONVOLUTIONAL MODULE #
#########################################

class GenConvPertModule(object):
    """
    Test module.
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, rand_chans, filt_shape,
                 rand_shape=None,
                 use_rand=True, use_conv=True, us_stride=2,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu',
                 mod_name='gm_conv'):
        assert ((us_stride == 1)), \
            "us_stride must be 1."
        assert ((in_chans == out_chans)), \
            "in_chans == out_chans is required."
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.rand_chans = rand_chans
        self.rand_shape = rand_shape
        self.filt_dim = filt_shape[0]
        self.use_rand = use_rand
        self.use_conv = use_conv
        self.us_stride = us_stride
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_bn_params = use_bn_params
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.in_chans + self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.conv_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.conv_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.conv_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize third conv layer parameters
        self.w3 = weight_ifn((self.out_chans, self.conv_chans, fd, fd),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        # derp a derp parameterrrrr
        self.wx = weight_ifn((self.conv_chans, self.rand_chans, 3, 3),
                             "{}_wx".format(self.mod_name))
        self.wy = weight_ifn((self.in_chans, self.conv_chans, 3, 3),
                             "{}_wy".format(self.mod_name))
        self.params.extend([self.wx, self.wy])
        return

    def share_params(self, source_module):
        """
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        """
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        # share third conv layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.g3, self.b3])
        # derp a derp parameterrrrr
        self.wx = source_module.wx
        self.wy = source_module.wy
        self.params.extend([self.wx, self.wy])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        self.wx.set_value(floatX(param_dict['wx']))
        self.wy.set_value(floatX(param_dict['wy']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        param_dict['wx'] = self.wx.get_value(borrow=False)
        param_dict['wy'] = self.wy.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False,
              share_mask=False, noise=None):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0]    # number of inputs in this batch
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        # augment input with random channels
        if rand_vals is None:
            if self.use_rand:
                # generate random values to append to module input
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0,
                                          dtype=theano.config.floatX)
            else:
                # FASTER THAN ALLOCATING 0s, WTF?
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                          dtype=theano.config.floatX)
        else:
            if not self.use_rand:
                # mask out random values, so they won't get used
                rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape  # return vals must be theano vars

        pert_input = T.concatenate([rand_vals, input], axis=1)
        # apply first internal conv layer
        h1 = dnn_conv(pert_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        # # apply second internal conv layer
        # h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        # if self.apply_bn:
        #     h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
        #                     use_gb=self.use_bn_params)
        # else:
        #     h2 = h2 + self.b2.dimshuffle('x',0,'x','x')
        #     h2 = add_noise(h2, noise=noise)
        # h2 = self.act_func(h2)
        # # apply final conv layer
        # h3 = dnn_conv(h2, self.w3, subsample=(1, 1), border_mode=(bm, bm))

        h3 = dnn_conv(h1, self.w3, subsample=(1, 1), border_mode=(bm, bm))

        # combine non-linear and linear transforms of input...
        h4 = input + h3
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b3.dimshuffle('x', 0, 'x', 'x')
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
        # output = h4
        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


#########################################
# DOUBLE GENERATOR CONVOLUTIONAL MODULE #
#########################################

class GenConvGRUModule(object):
    """
    Test module.
    """
    def __init__(self,
                 in_chans, out_chans, rand_chans, filt_shape,
                 rand_shape=None, use_rand=True,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu',
                 mod_name='gm_conv'):
        assert ((in_chans == out_chans)), \
            "in_chans == out_chans is required."
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.rand_chans = rand_chans
        self.filt_dim = filt_shape[0]
        self.rand_shape = rand_shape
        self.use_rand = use_rand
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_bn_params = use_bn_params
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize gate layer parameters
        self.w1 = weight_ifn((self.in_chans, (self.in_chans + self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.in_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.in_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])

        # initialize gate layer parameters
        self.w2 = weight_ifn((self.in_chans, (self.in_chans + self.rand_chans), fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.in_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.in_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])

        # initialize first new state layer parameters
        self.w3 = weight_ifn((self.in_chans, (self.in_chans + self.rand_chans), fd, fd),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.in_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.in_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def share_params(self, source_module):
        """
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        """
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        # share second conv layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False,
              share_mask=False, noise=None):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0]    # number of inputs in this batch
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans, input.shape[2], input.shape[3])
        # augment input with random channels
        if rand_vals is None:
            if self.use_rand:
                # generate random values to append to module input
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0,
                                          dtype=theano.config.floatX)
            else:
                # FASTER THAN ALLOCATING 0s, WTF?
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                          dtype=theano.config.floatX)
        else:
            if not self.use_rand:
                # mask out random values, so they won't get used
                rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape  # return vals must be theano vars

        # compute update gate and remember gate
        gate_input = T.concatenate([rand_vals, input], axis=1)
        h1 = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
            h1 = add_noise(h1, noise=noise)
        u = sigmoid(h1 + 1.)
        #
        h2 = dnn_conv(gate_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2.dimshuffle('x', 0, 'x', 'x')
            h2 = add_noise(h2, noise=noise)
        r = sigmoid(h2 + 1.)
        # compute new state proposal -- include hidden layer
        state_input = T.concatenate([rand_vals, r * input], axis=1)
        s = dnn_conv(state_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            s = switchy_bn(s, g=self.g2, b=self.b2, n=noise,
                           use_gb=self.use_bn_params)
        else:
            s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
            s = add_noise(s, noise=noise)
        s = self.act_func(s)
        # combine initial state and proposed new state based on u
        output = (u * input) + ((1. - u) * s)
        #
        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


###########################################
# GENERATOR DOUBLE FULLY-CONNECTED MODULE #
###########################################

class GenFCPertModule(object):
    """
    Residual-type module for fully-connected layers in the top-down
    pass for a hierarchical generative model.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        rand_chans: number of random channels to augment input
        fc_chans: number of channels in the "internal" layer
        use_rand: flag for whether or not to augment inputs
        use_fc: flag for whether to use "internal" layer
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to apply batch normalization
        use_bn_params: whether to use BN params
        act_func: ---
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, rand_chans, fc_chans,
                 rand_shape=None, use_rand=True, use_fc=True,
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu',
                 mod_name='gm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.rand_chans = rand_chans
        self.fc_chans = fc_chans
        self.rand_shape = rand_shape
        self.use_rand = use_rand
        self.use_fc = use_fc
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.use_bn_params = use_bn_params
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first conv layer parameters
        self.w1 = weight_ifn(((self.in_chans+self.rand_chans), self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.fc_chans, self.out_chans),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize convolutional projection layer parameters
        self.w3 = weight_ifn(((self.in_chans+self.rand_chans), self.out_chans),
                                "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        # derp a derp parameter
        self.wx = weight_ifn((self.rand_chans, self.in_chans),
                                "{}_wx".format(self.mod_name))
        self._w_ = weight_ifn((self.in_chans, self.in_chans),
                                "{}_w_".format(self.mod_name))
        self.params.extend([self.wx])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        self.wx.set_value(floatX(param_dict['wx']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        param_dict['wx'] = self.wx.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False,
              share_mask=False, noise=None):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0]    # number of inputs in this batch

        # get shape for random values that will augment input
        rand_shape = (batch_size, self.rand_chans)
        # augment input with random channels
        if rand_vals is None:
            if self.use_rand:
                # generate random values to append to module input
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0,
                                          dtype=theano.config.floatX)
            else:
                # FASTER THAN ALLOCATING 0s, WTF?
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=0.01,
                                          dtype=theano.config.floatX)
        else:
            if not self.use_rand:
                # mask out random values, so they won't get used
                rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape # return vals must be theano vars

        # apply dropout to input
        input = fc_drop_func(input, self.unif_drop, share_mask=share_mask)

        # perturb top-down input with a simple function of latent variables
        pert_vals = T.dot(rand_vals, self.wx)
        input = self.act_func( input + pert_vals )

        # stack random values on top of input
        full_input = T.concatenate([0.*rand_vals, input], axis=1)

        if self.use_fc:
            # apply first internal fc layer
            h1 = T.dot(full_input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # apply second internal fc layer
            h2 = T.dot(h1, self.w2)
            if self.apply_bn:
                h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h2 = h2 + self.b2.dimshuffle('x',0)
                h2 = add_noise(h2, noise=noise)
            # apply direct short-cut layer
            h3 = T.dot(full_input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + input #h3
        else:
            # only apply direct short-cut layer
            h4 = T.dot(full_input, self.w3)
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b3.dimshuffle('x',0)
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)

        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result


#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class InfConvGRUModuleIMS(object):
    """
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        im_chans: number of channels in the "info-merge" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        act_func: ---
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to condition on TD info
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, im_chans, rand_chans,
                 rand_shape=None,
                 act_func='tanh',
                 unif_drop=0.0,
                 chan_drop=0.0,
                 apply_bn=False,
                 use_td_cond=False,
                 use_bn_params=True,
                 unif_post=None,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
        self.rand_chans = rand_chans
        self.rand_shape = rand_shape
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = weight_ifn((self.im_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.im_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.im_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        #
        self.w2_im = weight_ifn((self.im_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.im_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.im_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize GRU state update parameters
        self.w3_im = weight_ifn((self.im_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((self.im_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((self.im_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # initialize conditioning parameters
        self.w4_im = weight_ifn((2*self.rand_chans, self.im_chans, 3, 3),
                                "{}_w4_im".format(self.mod_name))
        self.g4_im = gain_ifn((2*self.rand_chans), "{}_g4_im".format(self.mod_name))
        self.b4_im = bias_ifn((2*self.rand_chans), "{}_b4_im".format(self.mod_name))
        self.params.extend([self.w4_im, self.g4_im, self.b4_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((2*self.rand_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((2*self.rand_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((2*self.rand_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = source_module.w1_im
        self.g1_im = source_module.g1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        #
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # initialize conditioning parameters
        self.w4_im = source_module.w4_im
        self.g4_im = source_module.g4_im
        self.b4_im = source_module.b4_im
        self.params.extend([self.w4_im, self.g4_im, self.b4_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.g2_im.set_value(floatX(param_dict['g2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.g3_im.set_value(floatX(param_dict['g3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        self.w4_im.set_value(floatX(param_dict['w4_im']))
        self.g4_im.set_value(floatX(param_dict['g4_im']))
        self.b4_im.set_value(floatX(param_dict['b4_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['g2_im'] = self.g2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['g3_im'] = self.g3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        param_dict['w4_im'] = self.w4_im.get_value(borrow=False)
        param_dict['g4_im'] = self.g4_im.get_value(borrow=False)
        param_dict['b4_im'] = self.b4_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            # simple linear conditioning on top-down state
            h1 = dnn_conv(td_input, self.w1_td, subsample=(1,1), border_mode=(1,1))
            h1 = h1 + self.b1_td.dimshuffle('x',0,'x','x')
            out_mean = h1[:,:self.rand_chans,:,:]
            out_logvar = h1[:,self.rand_chans:,:,:]
        else:
            batch_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            rand_shape = (batch_size, self.rand_chans, rows, cols)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None, share_mask=False, noise=None):
        """
        Combine td_input, bu_input, and im_input to compute stuff.
        """
        # allocate a dummy im_input if None was provided
        if im_input is None:
            b_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            im_input = T.alloc(0.0, b_size, self.im_chans, rows, cols)

        # compute gating information for GRU state update
        gate_input = T.concatenate([td_input, bu_input, im_input], axis=1)
        u = dnn_conv(gate_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            u = switchy_bn(u, g=self.g1_im, b=self.b1_im, n=noise,
                           use_gb=self.use_bn_params)
        else:
            u = u + self.b1_im.dimshuffle('x',0,'x','x')
            u = add_noise(u, noise=noise)
        u = sigmoid(u + 1.)
        #
        r = dnn_conv(gate_input, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            r = switchy_bn(r, g=self.g2_im, b=self.b2_im, n=noise,
                           use_gb=self.use_bn_params)
        else:
            r = r + self.b2_im.dimshuffle('x',0,'x','x')
            r = add_noise(r, noise=noise)
        r = sigmoid(r + 1.)

        # compute new state for GRU state update
        state_input = T.concatenate([td_input, bu_input, r*im_input], axis=1)
        s = dnn_conv(state_input, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            s = switchy_bn(s, g=self.g3_im, b=self.b3_im, n=noise,
                           use_gb=self.use_bn_params)
        else:
            s = s + self.b3_im.dimshuffle('x',0,'x','x')
            s = add_noise(s, noise=noise)
        s = self.act_func(s)
        # combine initial state and proposed new state based on u
        out_im = (u * im_input) + ((1. - u) * s)

        # compute conditioning parameters
        h4 = dnn_conv(out_im, self.w4_im, subsample=(1, 1), border_mode=(1, 1))
        h4 = h4 + self.b4_im.dimshuffle('x',0,'x','x')
        out_mean = h4[:,:self.rand_chans,:,:]
        out_logvar = h4[:,self.rand_chans:,:,:]
        return out_mean, out_logvar, out_im


####################################
# BASIC FULLY-CONNECTED GRU MODULE #
####################################

class InfFCGRUModuleIMS(object):
    """
    GRU-type module that optionally takes an exogenous input.

    Params:
        td_chans: dimension of top-down input
        bu_chans: dimension of bottom-up input
        im_chans: dimension of inference state (i.e. the GRU)
        rand_chans: dimension of vars to put a conditional on
        act_func: --
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to use top-down conditioning
        use_bn_params: whether to use post-processing params for BN
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, im_chans, rand_chans,
                 rand_shape=None,
                 act_func='relu', unif_drop=0.0, apply_bn=True,
                 use_td_cond=False, use_bn_params=True,
                 unif_post=None,
                 mod_name='basic_fc_gru'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
        self.rand_chans = rand_chans
        self.rand_shape = rand_shape
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.mod_name = mod_name
        self.use_bn_params = use_bn_params
        self.unif_post = unif_post
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize gating parameters
        all_in_chans = self.td_chans + self.bu_chans + self.im_chans
        self.w1 = weight_ifn((all_in_chans, 2*self.im_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((2*self.im_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((2*self.im_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize state update parameters
        self.w2 = weight_ifn((all_in_chans, self.im_chans),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.im_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.im_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])
        # initialize conditional distribution parameters
        self.w3 = weight_ifn((self.im_chans, 2*self.rand_chans),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((2*self.rand_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((2*self.rand_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.w3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.g3.set_value(floatX(param_dict['g3']))
        self.b3.set_value(floatX(param_dict['b3']))
        return

    def share_params(self, source_module):
        """
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        """
        self.params = []
        # share gating layer parameters
        self.w1 = source_module.w1
        self.g1 = source_module.g1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.g1, self.b1])
        # share update layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.g2, self.b2])
        # share conditioning layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.g3, self.b3])
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['g3'] = self.g3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        batch_size = td_input.shape[0]
        rand_shape = (batch_size, self.rand_chans)
        out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                 dtype=theano.config.floatX)
        out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                   dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None, share_mask=False, noise=None):
        """
        Combine td_input, bu_input, and im_input to compute stuff.
        """
        # allocate a dummy im_input if None was provided
        if im_input is None:
            b_size = td_input.shape[0]
            im_input = T.alloc(0.0, b_size, self.im_chans)

        # stack up inputs to the gating functions
        g_in = T.concatenate([im_input, td_input, bu_input], axis=1)

        # compute gate stuff
        h1 = T.dot(g_in, self.w1)
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = add_noise(h1, noise=noise)
        h1 = sigmoid(h1 + 1.)
        # split information for update/recall gates
        r = h1[:,:self.im_chans]
        z = h1[:,self.im_chans:]

        # apply recall gate to input and compute state update proposal
        u_in = T.concatenate([r * im_input, td_input, bu_input], axis=1)
        h2 = T.dot(u_in, self.w2)
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2.dimshuffle('x',0)
            h2 = add_noise(h2, noise=noise)
        im_update = self.act_func(h2)

        # compute the updated GRU state
        out_im = (z * im_input) + ((1. - z) * im_update)

        # compute the conditional distribution from the new GRU state
        h3 = T.dot(out_im, self.w3)
        h3 = h3 + self.b3.dimshuffle('x', 0)
        out_mean = h3[:, :self.rand_chans]
        out_logvar = h3[:, self.rand_chans:]
        return out_mean, out_logvar, out_im


#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class InfConvMergeModuleIMS(object):
    """
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        im_chans: number of channels in the "info-merge" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        conv_chans: number of channels in the "internal" convolution layer
        use_conv: flag for whether to use "internal" convolution layer
        act_func: ---
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to condition on TD info
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, im_chans, rand_chans, conv_chans,
                 rand_shape=None,
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 unif_post=None,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
        self.rand_shape = rand_shape
        self.use_conv = use_conv
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.mod_type = mod_type
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters (from input -> hidden layer)
        if self.mod_type == 0:
            self.w1_im = weight_ifn((self.conv_chans, (self.td_chans + self.bu_chans + self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        else:
            self.w1_im = weight_ifn((self.conv_chans, (3 * self.td_chans + self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.conv_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second conv layer parameters (from hidden layer -> IM state perturbation)
        self.w2_im = weight_ifn((self.im_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.im_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.im_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize convolutional projection layer parameters
        self.w3_im = weight_ifn((2 * self.rand_chans, self.im_chans, 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((2 * self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2 * self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2 * self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters
        self.w1_im = source_module.w1_im
        self.g1_im = source_module.g1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = source_module.w2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.g2_im.set_value(floatX(param_dict['g2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.g3_im.set_value(floatX(param_dict['g3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['g2_im'] = self.g2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['g3_im'] = self.g3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            h1 = dnn_conv(td_input, self.w1_td, subsample=(1, 1), border_mode=(1, 1))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_td, b=self.b1_td, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_td.dimshuffle('x', 0, 'x', 'x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h2 = dnn_conv(h1, self.w2_td, subsample=(1, 1), border_mode=(1, 1))
            h3 = h2 + self.b2_td.dimshuffle('x', 0, 'x', 'x')
            out_mean = h3[:, :self.rand_chans, :, :]
            out_logvar = 0.0 * h3[:, self.rand_chans:, :, :]
        else:
            batch_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            rand_shape = (batch_size, self.rand_chans, rows, cols)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None, share_mask=False, noise=None):
        """
        Combine td_input, bu_input, and im_input to compute stuff.
        """
        # allocate a dummy im_input if None was provided
        if im_input is None:
            b_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            T.alloc(0.0, b_size, self.im_chans, rows, cols)

        # stack top-down and bottom-up inputs on top of each other
        if self.mod_type == 0:
            full_input = T.concatenate([td_input, bu_input, im_input], axis=1)
        else:
            full_input = T.concatenate([td_input, bu_input, td_input - bu_input, im_input], axis=1)
        # do dropout
        full_input = conv_drop_func(full_input, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)

        # apply first internal conv layer
        h1 = dnn_conv(full_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1_im, b=self.b1_im, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1_im.dimshuffle('x', 0, 'x', 'x')
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                            share_mask=share_mask)
        # apply second internal conv layer
        h2 = dnn_conv(h1, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2_im, b=self.b2_im, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2_im.dimshuffle('x', 0, 'x', 'x')
            h2 = add_noise(h2, noise=noise)

        # apply perturbation to IM input, then apply non-linearity
        out_im = self.act_func(im_input + h2)

        # compute conditional parameters from the updated IM state
        h3 = dnn_conv(out_im, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
        h3 = h3 + self.b3_im.dimshuffle('x', 0, 'x', 'x')
        out_mean = h3[:, :self.rand_chans, :, :]
        out_logvar = h3[:, self.rand_chans:, :, :]
        return out_mean, out_logvar, out_im

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
        im_chans: number of channels in the "IM state" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        conv_chans: number of channels in the "internal" convolution layer
        use_conv: flag for whether to use "internal" convolution layer
        act_func: ---
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to condition on TD info
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, im_chans, rand_chans, conv_chans,
                 rand_shape=None,
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 unif_post=None,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
        self.rand_shape = rand_shape
        self.use_conv = use_conv
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.mod_type = mod_type
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters
        if self.mod_type == 0:
            self.w1_im = weight_ifn((self.conv_chans, (self.td_chans+self.bu_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        else:
            self.w1_im = weight_ifn((self.conv_chans, (3*self.td_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.conv_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.params.extend([self.w2_im])
        # initialize convolutional projection layer parameters
        if self.mod_type == 0:
            # module acts just on TD and BU input
            self.w3_im = weight_ifn((2*self.rand_chans, (self.td_chans+self.bu_chans), 3, 3),
                                    "{}_w3_im".format(self.mod_name))
        else:
            # module acts on TD and BU input, and their difference
            self.w3_im = weight_ifn((2*self.rand_chans, (3*self.td_chans), 3, 3),
                                    "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2*self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters
        self.w1_im = source_module.w1_im
        self.g1_im = source_module.g1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = source_module.w2_im
        self.params.extend([self.w2_im])
        # module acts just on TD and BU input
        self.w3_im = source_module.w3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = source_module.w2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            h1 = dnn_conv(td_input, self.w1_td, subsample=(1,1), border_mode=(1,1))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_td, b=self.b1_td, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_td.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h2 = dnn_conv(h1, self.w2_td, subsample=(1,1), border_mode=(1,1))
            h3 = h2 + self.b2_td.dimshuffle('x',0,'x','x')
            out_mean = h3[:,:self.rand_chans,:,:]
            out_logvar = 0.0 * h3[:,self.rand_chans:,:,:] # use fixed logvar...
        else:
            batch_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            rand_shape = (batch_size, self.rand_chans, rows, cols)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None, share_mask=False, noise=None):
        """
        Combine td_input and bu_input, to put distributions over some stuff.
        """
        # stack top-down and bottom-up inputs on top of each other
        if self.mod_type == 0:
            full_input = T.concatenate([td_input, bu_input], axis=1)
        else:
            full_input = T.concatenate([td_input, bu_input, td_input - bu_input], axis=1)
        # do dropout
        full_input = conv_drop_func(full_input, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(full_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_im, b=self.b1_im, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_im.dimshuffle('x', 0, 'x', 'x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
            # apply direct short-cut conv layer
            h3 = dnn_conv(full_input, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
            # combine non-linear and linear transforms of input...
            h4 = h2 + self.b3_im.dimshuffle('x', 0, 'x', 'x')
        else:
            # apply direct short-cut conv layer
            h3 = dnn_conv(full_input, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
            h4 = h3 + self.b3_im.dimshuffle('x', 0, 'x', 'x')
        # split output into "mean" and "log variance" components, for using in
        # Gaussian reparametrization.
        out_mean = h4[:, :self.rand_chans, :, :]
        out_logvar = h4[:, self.rand_chans:, :, :]
        return out_mean, out_logvar, None

#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class InfFCMergeModuleIMS(object):
    """
    Module for merging bottom-up and top-down information in a deep generative
    network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        im_chans: number of channels in the "info-merge" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        fc_chans: number of channels in the "internal" layer
        use_fc: flag for whether to use "internal" layer
        act_func: ---
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to condition on TD info
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, im_chans, fc_chans, rand_chans,
                 rand_shape=None,
                 use_fc=True, act_func='relu',
                 unif_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 unif_post=None,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
        self.rand_chans = rand_chans
        self.fc_chans = fc_chans
        self.rand_shape = rand_shape
        self.use_fc = use_fc
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.unif_drop = unif_drop
        self.mod_type = mod_type
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first layer parameters (from input -> hidden layer)
        if self.mod_type == 0:
            self.w1_im = weight_ifn(((self.td_chans + self.bu_chans + self.im_chans), self.fc_chans),
                                    "{}_w1_im".format(self.mod_name))
        else:
            self.w1_im = weight_ifn(((3 * self.td_chans + self.im_chans), self.fc_chans),
                                    "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.fc_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.fc_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second layer parameters (from hidden layer -> IM state perturbation)
        self.w2_im = weight_ifn((self.fc_chans, self.im_chans),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.im_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.im_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = weight_ifn((self.im_chans, 2 * self.rand_chans),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((2 * self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.td_chans, self.fc_chans),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.fc_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.fc_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((self.fc_chans, 2 * self.rand_chans),
                                    "{}_w2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2 * self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first layer parameters
        self.w1_im = source_module.w1_im
        self.g1_im = source_module.g1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second layer parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second layer parameters
            self.w2_td = source_module.w2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.g2_im.set_value(floatX(param_dict['g2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.g3_im.set_value(floatX(param_dict['g3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['g2_im'] = self.g2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['g3_im'] = self.g3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            h1 = T.dot(td_input, self.w1_td)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_td, b=self.b1_td, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_td.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h2 = T.dot(h1, self.w2_td)
            h3 = h2 + self.b2_td.dimshuffle('x', 0)
            out_mean = h3[:, :self.rand_chans]
            out_logvar = 0.0 * h3[:, self.rand_chans:]
        else:
            batch_size = td_input.shape[0]
            rand_shape = (batch_size, self.rand_chans)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None, share_mask=False, noise=None):
        """
        Combine td_input, bu_input, and im_input to compute stuff.
        """
        # allocate a dummy im_input if None was provided
        if im_input is None:
            b_size = td_input.shape[0]
            T.alloc(0.0, b_size, self.im_chans)
        # stack top-down and bottom-up inputs
        if self.mod_type == 0:
            full_input = T.concatenate([td_input, bu_input, im_input], axis=1)
        else:
            full_input = T.concatenate([td_input, bu_input, td_input - bu_input, im_input], axis=1)
        # do dropout
        full_input = fc_drop_func(full_input, self.unif_drop, share_mask=share_mask)

        # apply first internal layer
        h1 = T.dot(full_input, self.w1_im)
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1_im, b=self.b1_im, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1_im.dimshuffle('x', 0)
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
        # apply second internal layer
        h2 = T.dot(h1, self.w2_im)
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2_im, b=self.b2_im, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2_im.dimshuffle('x', 0)
            h2 = add_noise(h2, noise=noise)

        # apply perturbation to IM input, then apply non-linearity
        out_im = self.act_func(im_input + h2)

        # compute conditional parameters from the updated IM state
        h3 = T.dot(out_im, self.w3_im)
        h3 = h3 + self.b3_im.dimshuffle('x', 0)
        out_mean = h3[:, :self.rand_chans]
        out_logvar = h3[:, self.rand_chans:]
        return out_mean, out_logvar, out_im


####################################
# INFERENCE FULLY CONNECTED MODULE #
####################################

class InfFCMergeModule(object):
    """
    Module that feeds forward through a single fully connected hidden layer
    and then produces a conditional over some Gaussian latent variables.

    Params:
        td_chans: dimension of the "top-down" inputs
        bu_chans: dimension of the "bottom-up" inputs to the module
        fc_chans: dimension of the fully connected layer
        rand_chans: dimension of the Gaussian latent vars of interest
        use_fc: flag for whether to use the hidden fully connected layer
        use_sc: flag for whether to include linear shortcut conditioning
        act_func: ---
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to use batch normalization
        use_td_cond: whether to use top-down conditioning
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, td_chans, bu_chans, fc_chans, rand_chans,
                 rand_shape=None,
                 use_fc=True, use_sc=True, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 unif_post=None,
                 mod_name='im_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.fc_chans = fc_chans
        self.rand_chans = rand_chans
        self.rand_shape = rand_shape
        self.use_fc = use_fc
        self.use_sc = use_sc
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first layer parameters (from input -> hidden layer)
        self.w1_im = weight_ifn(((self.td_chans + self.bu_chans), self.fc_chans),
                                "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.fc_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.fc_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second layer parameters (from hidden layer -> IM state perturbation)
        self.w2_im = weight_ifn((self.fc_chans, 2 * self.rand_chans),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((2 * self.rand_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((2 * self.rand_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = weight_ifn(((self.td_chans + self.bu_chans), 2 * self.rand_chans),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((2 * self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.td_chans, self.fc_chans),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.fc_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.fc_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((self.fc_chans, 2 * self.rand_chans),
                                    "{}_w2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2 * self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
        return

    def share_params(self, source_module):
        """
        Set this module to share parameters with source_module.
        """
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first layer parameters
        self.w1_im = source_module.w1_im
        self.g1_im = source_module.g1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second layer parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second layer parameters
            self.w2_td = source_module.w2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.g2_im.set_value(floatX(param_dict['g2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.g3_im.set_value(floatX(param_dict['g3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['g2_im'] = self.g2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['g3_im'] = self.g3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            h1 = T.dot(td_input, self.w1_td)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_td, b=self.b1_td, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_td.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h2 = T.dot(h1, self.w2_td)
            h3 = h2 + self.b2_td.dimshuffle('x', 0)
            out_mean = h3[:, :self.rand_chans]
            out_logvar = 0.0 * h3[:, self.rand_chans:]
        else:
            batch_size = td_input.shape[0]
            rand_shape = (batch_size, self.rand_chans)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, im_input=None,
                 share_mask=False, noise=None):
        """
        Apply this fully connected inference module to the given input. This
        produces a set of means and log variances for some Gaussian variables.
        """
        # flatten input to 1d per example
        td_input = T.flatten(td_input, 2)
        bu_input = T.flatten(bu_input, 2)
        # concatenate TD and BU inputs
        full_input = T.concatenate([td_input, bu_input], axis=1)

        # apply dropout
        full_input = fc_drop_func(full_input, self.unif_drop,
                                  share_mask=share_mask)
        if self.use_fc:
            # feedforward to fc layer
            h1 = T.dot(full_input, self.w1_im)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1_im, b=self.b1_im, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1_im.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward from fc layer to output
            h2 = T.dot(h1, self.w2_im)
            # feedforward directly from BU/TD inputs to output
            h3 = T.dot(full_input, self.w3_im)
            if self.use_sc:
                h4 = h2 + self.b3_im.dimshuffle('x', 0) + h3
            else:
                h4 = h2 + self.b3_im.dimshuffle('x', 0)
        else:
            # feedforward directly from BU input to output
            h3 = T.dot(full_input, self.w3_im)
            h4 = h3 + self.b3_im.dimshuffle('x', 0)
        # split output into mean and log variance parts
        out_mean = h4[:, :self.rand_chans]
        out_logvar = h4[:, self.rand_chans:]
        return out_mean, out_logvar, None


#######################################################
# INFERENCE FULLY CONNECTED MODULE FOR TOP OF NETWORK #
#######################################################

class InfTopModule(object):
    """
    Module that feeds forward through a single fully connected hidden layer
    and then produces a conditional over some Gaussian latent variables.

    Params:
        bu_chans: dimension of the "bottom-up" inputs to the module
        fc_chans: dimension of the fully connected layer
        rand_chans: dimension of the Gaussian latent vars of interest
        use_fc: flag for whether to use the hidden fully connected layer
        use_sc: flag for whether to include linear shortcut conditioning
        act_func: ---
        unif_drop: drop rate for unifor dropout
        apply_bn: whether to use batch normalization
        use_bn_params: whether to use BN params
        unif_post: optional param for using uniform reparametrization with
                   bounded posterior KL.
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, bu_chans, fc_chans, rand_chans,
                 rand_shape=None,
                 use_fc=True, use_sc=False, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True,
                 unif_post=None,
                 mod_name='dm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.bu_chans = bu_chans
        self.fc_chans = fc_chans
        self.rand_chans = rand_chans
        self.rand_shape = rand_shape
        self.use_fc = use_fc
        self.use_sc = use_sc
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.unif_drop = unif_drop
        self.apply_bn = apply_bn
        self.use_bn_params = True
        self.unif_post = unif_post
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize weights for transform into fc layer
        self.w1 = weight_ifn((self.bu_chans, self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        # initialize weights for transform out of fc layer
        self.w2 = weight_ifn((self.fc_chans, 2 * self.rand_chans),
                             "{}_w2".format(self.mod_name))
        self.params.extend([self.w2])
        # initialize weights for transform straight from input to output
        self.w3 = weight_ifn((self.bu_chans, 2 * self.rand_chans),
                             "{}_w3".format(self.mod_name))
        self.b3 = bias_ifn((2 * self.rand_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.b3.set_value(floatX(param_dict['b3']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply(self, bu_input, share_mask=False, noise=None):
        """
        Apply this fully connected inference module to the given input. This
        produces a set of means and log variances for some Gaussian variables.
        """
        # flatten input to 1d per example
        bu_input = T.flatten(bu_input, 2)
        # apply dropout
        bu_input = fc_drop_func(bu_input, self.unif_drop,
                                share_mask=share_mask)
        if self.use_fc:
            # feedforward to fc layer
            h1 = T.dot(bu_input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x', 0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward from fc layer to output
            h2 = T.dot(h1, self.w2)
            # feedforward directly from BU input to output
            h3 = T.dot(bu_input, self.w3)
            if self.use_sc:
                h4 = h2 + self.b3.dimshuffle('x', 0) + h3
            else:
                h4 = h2 + self.b3.dimshuffle('x', 0)
        else:
            # feedforward directly from BU input to output
            h3 = T.dot(bu_input, self.w3)
            h4 = h3 + self.b3.dimshuffle('x', 0)
        # split output into mean and log variance parts
        out_mean = h4[:, :self.rand_chans]
        out_logvar = h4[:, self.rand_chans:]
        return out_mean, out_logvar


#####################################
# Simple MLP fully connected module #
#####################################

class MlpFCModule(object):
    """
    Module that transforms values through a single fully connected layer.
    """
    def __init__(self,
                 in_dim, out_dim,
                 apply_bn=True,
                 unif_drop=0.0,
                 act_func='relu',
                 use_bn_params=True,
                 mod_name='dm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.apply_bn = apply_bn
        self.unif_drop = unif_drop
        self.use_bn_params = use_bn_params
        if act_func == 'ident':
            self.act_func = lambda x: x
        elif act_func == 'tanh':
            self.act_func = lambda x: tanh(x)
        elif act_func == 'elu':
            self.act_func = lambda x: elu(x)
        elif act_func == 'relu':
            self.act_func = lambda x: relu(x)
        else:
            self.act_func = lambda x: lrelu(x)
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Orthogonal()
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        # initialize first layer parameters
        self.w1 = weight_ifn((self.in_dim, self.out_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.out_dim), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_dim), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input, share_mask=False, noise=None):
        """
        Apply this fully-connected module.
        """
        # flatten input to 1d per example
        h1 = T.flatten(input, 2)
        # apply dropout
        h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
        # feed-forward through layer
        h2 = T.dot(h1, self.w1)
        if self.apply_bn:
            h3 = switchy_bn(h2, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h3 = h2 + self.b1.dimshuffle('x', 0)
            h3 = add_noise(h3, noise=noise)
        h4 = self.act_func(h3)
        return h4


class ClassConvModule(object):
    '''
    Simple convolutional layer for use anywhere?

    Params:
        in_chans: number of channels in input
        class_count: number of classes to form predictions over
        filt_shape: filter shape, should be square and odd dim
        bu_source: BU module that feeds into this multi-local class predictor
        stride: whether to use 'double', 'single', or 'half' stride.
        act_func: --
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        use_noise: whether to use provided noise during apply()
        mod_name: text name to identify this module in theano graph
    '''
    def __init__(self, in_chans, class_count, filt_shape, bu_source,
                 stride='single', act_func='ident',
                 unif_drop=0.0, chan_drop=0.0,
                 use_noise=True,
                 mod_name='class_conv'):
        assert ((filt_shape[0] % 2) > 0), "filter dim should be odd (not even)"
        assert (stride in ['single', 'double', 'half']), \
            "stride should be 'single', 'double', or 'half'."
        self.in_chans = in_chans
        self.class_count = class_count
        self.filt_dim = filt_shape[0]
        self.bu_source = bu_source
        self.stride = stride
        self.unif_drop = unif_drop
        self.chan_drop = chan_drop
        self.use_noise = use_noise
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.class_count, self.in_chans, self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((self.class_count), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.b1]
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input, noise=None):
        """
        Apply this convolutional module to the given input.
        """
        noise = noise if self.use_noise else None
        bm = int((self.filt_dim - 1) / 2)  # use "same" mode convolutions
        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop)
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
        h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
        # get class predictions via channel-wise max pooling (over all space)
        class_preds = T.max(T.max(h1, axis=2), axis=2)
        return class_preds


class GMMPriorModule(object):
    '''
    Class for managing a Gaussian Mixture Model prior over the top-most latent
    variables in a deep generative model.
    '''
    def __init__(self, mix_comps, mix_dim, shared_dim=None, mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.mix_comps = mix_comps
        self.mix_dim = mix_dim
        self.shared_dim = shared_dim
        # make a (non-trainable) mask to force some dimensions to be shared
        smask = np.ones((self.mix_dim,))
        if (self.shared_dim is not None) and (self.shared_dim > 0):
            smask[:self.shared_dim] = 0.  # block inter-component variation on shared dims
        self.shared_mask = sharedX(smask)
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.1)
        self.M = weight_ifn((self.mix_dim, self.mix_comps),
                            "{}_M".format(self.mod_name))
        self.V = weight_ifn((self.mix_dim, self.mix_comps),
                            "{}_V".format(self.mod_name))
        self.params = [self.M, self.V]
        return

    def load_params(self, param_dict):
        """
        Load module params directly from a dict of numpy arrays.
        """
        self.M.set_value(floatX(param_dict['M']))
        self.V.set_value(floatX(param_dict['V']))
        return

    def dump_params(self):
        """
        Dump module params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['M'] = self.M.get_value(borrow=False)
        param_dict['V'] = self.V.get_value(borrow=False)
        return param_dict

    def shuffle_components(self):
        """
        Shuffle order of mixture components.
        """
        comp_idx = np.arange(self.mix_comps)
        npr.shuffle(comp_idx)
        _M = self.M.get_value(borrow=False).T
        _V = self.V.get_value(borrow=False).T
        _M = _M[comp_idx]
        _V = _V[comp_idx]
        self.M.set_value(floatX(_M.T))
        self.V.set_value(floatX(_V.T))
        return

    def compute_kld_info(self, in_means, in_logvars, z_vals):
        '''
        Compute information about KL divergence between distributions with
        the given parameters and this GMM. Use z samples in z_vals.

        Note: all returned values are shape (n_batch,)
        '''
        #
        # use analytical approximation:
        # KL(q || p) ~ -log_mean_exp(-KL(q || p1), -KL(q || p2), ..., -KL(q || pN))
        #
        # we assume mixtures over '1D' latent vars, i.e. params are matrices
        #
        # for now, we assume uniform mixture component weights
        #
        C = -0.918939  # -log(2 * pi) / 2

        # compute elementwise vals at shape: (n_batch, mix_dim, mix_comps)
        M_in = in_means.dimshuffle(0, 1, 'x')
        V_in = in_logvars.dimshuffle(0, 1, 'x')
        Z_in = z_vals.dimshuffle(0, 1, 'x')
        M_mix = (self.M.dimshuffle('x', 0, 1) *
                 self.shared_mask.dimshuffle('x', 0, 'x'))
        V_mix = (self.V.dimshuffle('x', 0, 1) *
                 self.shared_mask.dimshuffle('x', 0, 'x'))
        # compute mix_comp_kld, with shape: (n_batch, mix_comps)
        mix_comp_kld = 0.5 * (V_mix - V_in +
                              (T.exp(V_in) / T.exp(V_mix)) +
                              ((M_in - M_mix)**2. / T.exp(V_mix)) - 1.)
        mix_comp_kld = T.sum(mix_comp_kld, axis=1)
        # compute KL(q || p) approximation
        mix_kld_apprx = -log_mean_exp(-mix_comp_kld, axis=1).flatten()

        # compute compute log p1(z), log p2(z) for each mixture component
        log_mix_z = C - (0.5 * V_mix) - \
            ((M_mix - Z_in)**2. / (2. * T.exp(V_mix)))
        log_mix_z = T.sum(log_mix_z, axis=1)
        # log_mix_z.shape: (n_batch, mix_comps)

        # compute log p(z) for full mixture using log-mean-exp
        log_p_z = log_mean_exp(log_mix_z, axis=1).flatten()

        # compute exact posteriors over mixture components
        ws_mat = log_mix_z - T.max(log_mix_z, axis=1, keepdims=True)
        ws_mat = T.exp(ws_mat)
        ws_mat = ws_mat / T.sum(ws_mat, axis=1, keepdims=True)
        # ws_mat_dcg = theano.gradient.disconnected_grad(ws_mat)
        # ws_mat.shape: (n_batch, mix_comps)

        # entropy of mixture posteriors    -- shape: (n_batch,)
        mix_post_ent = -T.sum(ws_mat * T.log(ws_mat + 1e-5), axis=1)
        # batch estimate of misture weight -- shape: (mix_comps,)
        mix_comp_weight = T.mean(ws_mat, axis=0)
        mix_comp_post = ws_mat

        # # marginalize free-energy over the true posterior, assuming a uniform
        # # prior over z. I.e., we assume uniform mixture weights
        # # -- we get E_{p(z|x)}[log p(x|z)] - KL(p(z|x) || p(z))
        # log_p_z = T.sum((ws_mat_dcg * log_mix_z), axis=1) - \
        #     T.sum(ws_mat * T.log(ws_mat + 1e-5), axis=1)

        # compute log q(z)
        log_q_z = C - (0.5 * in_logvars) - ((z_vals - in_means)**2. /
                                            (2. * T.exp(in_logvars)))
        log_q_z = T.sum(log_q_z, axis=1)
        return mix_kld_apprx, mix_comp_kld, mix_comp_post, log_p_z, log_q_z, mix_post_ent, mix_comp_weight

    def sample_mix_comps(self, comp_idx=None, batch_size=None):
        '''
        Sample either from a given sequence of mixture components, or from
        components sampled at random.

        Inputs:
            comp_idx: a sequence of component IDs to sample from.
            batch_size: number of independent samples to draw.
        '''
        M_np = self.M.get_value(borrow=False).T
        V_np = self.V.get_value(borrow=False).T
        if (comp_idx is None) and (batch_size is not None):
            comp_idx = npr.randint(low=0, high=M_np.shape[0], size=(batch_size,))
        # get means and log variances for samples
        M_samp = M_np[comp_idx, :]
        V_samp = V_np[comp_idx, :]
        # sample ZMUV Gaussian vars, then reparametrize
        z_zmuv = np_rng.normal(size=M_samp.shape)
        z_samp = M_samp + (np.exp(0.5 * V_samp) * z_zmuv)
        return floatX(z_samp)


class TDRefinerWrapper(object):
    '''
    Wrapper around a TD module and a "decoder" with one hidden layer.
    -- This is for "full-res refinement".

    Params:
        gen_module: the first module in this TD module to apply. Inputs are
                    a top-down input and a latent input.
        mlp_modules: a list of the modules to apply to the output of gen_module.
    '''
    def __init__(self, gen_module, mlp_modules=None, mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.gen_module = gen_module
        self.params = [p for p in gen_module.params]
        if mlp_modules is not None:
            # use some extra post-processing modules
            self.mlp_modules = [m for m in mlp_modules]
            for mlp_mod in self.mlp_modules:
                self.params.extend(mlp_mod.params)
        else:
            # don't use any extra post-processing modules
            self.mlp_modules = None
        self.mod_name = mod_name
        self.has_decoder = True
        return

    def dump_params(self):
        '''
        Dump params for later reloading by self.load_params.
        '''
        mod_param_dicts = [self.gen_module.dump_params()]
        if self.mlp_modules is not None:
            mod_param_dicts.extend([m.dump_params() for m in self.mlp_modules])
        return mod_param_dicts

    def load_params(self, param_dict=None):
        '''
        Load params from the output of self.dump_params.
        '''
        self.gen_module.load_params(param_dict=param_dict[0])
        if self.mlp_modules is not None:
            for pdict, mod in zip(param_dict[1:], self.mlp_modules):
                mod.load_params(param_dict=pdict)
        return

    def apply(self, input, rand_vals, noise=None):
        '''
        Process the gen_module, then apply self.mlp_modules.
        '''
        # apply the basic TD module
        h1 = self.gen_module.apply(input=input, rand_vals=rand_vals)
        # apply the "decoder"
        acts = None
        for mod in self.mlp_modules:
            if acts is None:
                acts = [mod.apply(h1)]
            else:
                acts.append(mod.apply(acts[-1]))
        h2 = acts[-1]
        return h1, h2


##############
# EYE BUFFER #
##############
