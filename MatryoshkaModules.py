import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, \
                    add_noise

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
            r = r.dimshuffle('x',0)
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
            rc = cu_rng.uniform((x.shape[0],x.shape[1]),
                                dtype=theano.config.floatX)
            chan_mask = (rc > chan_drop)
            x = x * chan_mask.dimshuffle(0,1,'x','x')
    else:
        # use the same mask for entire batch
        if unif_drop > 0.01:
            ru = cu_rng.uniform((x.shape[1],x.shape[2],x.shape[3]),
                                dtype=theano.config.floatX)
            ru = ru.dimshuffle('x',0,1,2)
            x = x * (ru > unif_drop)
        if chan_drop > 0.01:
            rc = cu_rng.uniform((x.shape[1],),
                                dtype=theano.config.floatX)
            chan_mask = (rc > chan_drop)
            x = x * chan_mask.dimshuffle('x',0,'x','x')
    return x

def switchy_bn(acts, g=None, b=None, use_gb=True, n=None):
    if use_gb and (not (g is None) or (b is None)):
        bn_acts = batchnorm(acts, g=g, b=b, n=n)
    else:
        bn_acts = batchnorm(acts, n=n)
    return bn_acts

#######################################
# BASIC DOUBLE FULLY-CONNECTED MODULE #
#######################################

class BasicFCResModule(object):
    """
    Module with a direct pass-through connection that gets modulated by a pair
    of hidden layers.

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
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
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
            # apply first internal conv layer (might downsample)
            h1 = T.dot(input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            # apply dropout at intermediate convolution layer
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # apply second internal conv layer
            h2 = T.dot(h1, self.w2)
            # apply pass-through conv layer (might downsample)
            h3 = T.dot(input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct pass-through connection
            h4 = T.dot(input, self.w3)
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b3.dimshuffle('x',0)
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
        if rand_shapes:
            result = [output, input.shape]
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
        apply_bn: whether to apply batch normalization after conv
        act_func: --
        unif_drop: drop rate for uniform dropout
        use_bn_params: whether to use params for BN
        use_noise: whether to use the provided noise durnig apply()
        mod_name: text name to identify this module in theano graph
    """
    def __init__(self, in_chans, out_chans,
                 apply_bn=True, act_func='ident',
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
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
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
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
        h1 = T.dot(input, self.w1)
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        if rand_shapes:
            result = [h1, input.shape]
        else:
            result = h1
        return result

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
        assert (stride in ['single', 'double', 'half']), \
                "stride must be 'double', 'single', or 'half'."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        assert (filt_shape == (3,3) or filt_shape == (5,5)), \
                "filt_shape must be (3,3) or (5,5)."
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
        self.w_prj = weight_ifn((self.out_chans, self.in_chans, fd, fd),
                                "{}_w_prj".format(self.mod_name))
        self.g_prj = gain_ifn((self.out_chans), "{}_g_prj".format(self.mod_name))
        self.b_prj = bias_ifn((self.out_chans), "{}_b_prj".format(self.mod_name))
        self.params.extend([self.w_prj, self.g_prj, self.b_prj])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w_prj.set_value(floatX(param_dict['w_prj']))
        self.g_prj.set_value(floatX(param_dict['g_prj']))
        self.b_prj.set_value(floatX(param_dict['b_prj']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w_prj'] = self.w_prj.get_value(borrow=False)
        param_dict['g_prj'] = self.g_prj.get_value(borrow=False)
        param_dict['b_prj'] = self.b_prj.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals=None, rand_shapes=False, noise=None,
              share_mask=False):
        """
        Apply this convolutional module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = 1 if (self.stride == 'single') else 2
        bm = (self.filt_dim - 1) // 2
        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)
        if self.use_conv:
            if self.stride in ['double', 'single']:
                # apply first internal conv layer (might downsample)
                h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
                if self.apply_bn:
                    h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                    use_gb=self.use_bn_params)
                else:
                    h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                    h1 = add_noise(h1, noise=noise)
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer
                h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
                # apply pass-through conv layer (might downsample)
                h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                # apply first internal conv layer
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
                if self.apply_bn:
                    h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                    use_gb=self.use_bn_params)
                else:
                    h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                    h1 = add_noise(h1, noise=noise)
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer (might upsample)
                h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
                # apply pass-through conv layer (might upsample)
                h3 = deconv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct pass-through conv layer
            if self.stride in ['double', 'single']:
                h4 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                h4 = deconv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g_prj, b=self.b_prj, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b_prj.dimshuffle('x',0,'x','x')
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
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
                 use_bn_params=True,
                 use_noise=True,
                 mod_name='basic_conv'):
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
        self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
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
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
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
            h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
            h1 = add_noise(h1, noise=noise)
        h1 = self.act_func(h1)
        if rand_shapes:
            result = [h1, input.shape]
        else:
            result = h1
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
                 apply_bn=True, init_func=None,
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this discriminator module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.02)
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
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.w3.set_value(floatX(param_dict['w3']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
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
            y = h2
        else:
            h2 = T.dot(input, self.w3)
            y = h2
        return [h2, y]


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
        assert (filt_shape == (3,3) or filt_shape == (5,5)), \
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
        self.w_prj = weight_ifn((self.out_chans, self.in_chans, fd, fd),
                                "{}_w_prj".format(self.mod_name))
        self.g_prj = gain_ifn((self.out_chans), "{}_g_prj".format(self.mod_name))
        self.b_prj = bias_ifn((self.out_chans), "{}_b_prj".format(self.mod_name))
        self.params.extend([self.w_prj, self.g_prj, self.b_prj])
        # initialize weights for the "discrimination" layer
        self.wd = weight_ifn((1, self.out_chans, 3, 3),
                             "{}_wd".format(self.mod_name))
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.g2.set_value(floatX(param_dict['g2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w_prj.set_value(floatX(param_dict['w_prj']))
        self.g_prj.set_value(floatX(param_dict['g_prj']))
        self.b_prj.set_value(floatX(param_dict['b_prj']))
        self.wd.set_value(floatX(param_dict['wd']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['g2'] = self.g2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w_prj'] = self.w_prj.get_value(borrow=False)
        param_dict['g_prj'] = self.g_prj.get_value(borrow=False)
        param_dict['b_prj'] = self.b_prj.get_value(borrow=False)
        param_dict['wd'] = self.wd.get_value(borrow=False)
        return param_dict

    def apply(self, input, noise=None, share_mask=False):
        """
        Apply this generator module to some input.
        """
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.ds_stride         # stride for "learned downsampling"
        bm = (self.filt_dim - 1) // 2 # set border mode for the convolutions
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
                h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            # apply dropout at intermediate convolution layer
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)

            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
            # apply direct input->output "projection" layer
            h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))

            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
            if self.apply_bn:
                h4 = switchy_bn(h4, g=self.g_prj, b=self.b_prj, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h4 = h4 + self.b_prj.dimshuffle('x',0,'x','x')
                h4 = add_noise(h4, noise=noise)
            output = self.act_func(h4)
        else:
            # apply direct input->output "projection" layer
            h3 = dnn_conv(input, self.w_prj, subsample=(ss, ss), border_mode=(bm, bm))
            if self.apply_bn:
                h3 = switchy_bn(h3, g=self.g_prj, b=self.b_prj, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h3 = h3 + self.b_prj.dimshuffle('x',0,'x','x')
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
    Module that transforms random values through a single fully connected
    layer, and then a linear transform (with another relu, optionally).
    """
    def __init__(self,
                 rand_dim, fc_dim, out_shape,
                 use_fc, apply_bn=True,
                 unif_drop=0.0,
                 act_func='relu',
                 use_bn_params=True,
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
        self.fc_dim = fc_dim
        self.use_fc = use_fc
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
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
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
            rand_shape = (batch_size, self.rand_dim)
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                      dtype=theano.config.floatX)
        else:
            rand_shape = (rand_vals.shape[0], self.rand_dim)
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape

        # drop from latent vars!
        # rand_vals = fc_drop_func(rand_vals, self.unif_drop,
        #                          share_mask=share_mask)

        if self.use_fc:
            h1 = T.dot(rand_vals, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            h2 = T.dot(h1, self.w2) + T.dot(rand_vals, self.w3)
        else:
            h2 = T.dot(rand_vals, self.w3)
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b3.dimshuffle('x',0)
            h2 = add_noise(h2, noise=noise)
        h2 = self.act_func(h2)
        if len(self.out_shape) > 1:
            # reshape vector outputs for use as conv layer inputs
            h2 = h2.reshape((h2.shape[0], self.out_shape[0], \
                             self.out_shape[1], self.out_shape[2]))
        if rand_shapes:
            result = [h2, rand_shape]
        else:
            result = h2
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
        filt_shape: size of filters (either (3, 3) or (5, 5))
        use_rand: flag for whether or not to augment inputs
        use_conv: flag for whether to use "internal" convolution layer
        us_stride: upsampling ratio in the fractionally strided convolution
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_bn_params: whether to use BN params
        act_func: ---
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 in_chans, out_chans, conv_chans, rand_chans, filt_shape,
                 use_rand=True, use_conv=True, us_stride=2,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu', mod_type=1,
                 mod_name='gm_conv'):
        assert ((us_stride == 1) or (us_stride == 2)), \
                "us_stride must be 1 or 2."
        assert (filt_shape == (3,3) or filt_shape == (5,5)), \
                "filt_shape must be (3,3) or (5,5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.rand_chans = rand_chans
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
        self.mod_type = mod_type
        # use small dummy rand size if we won't use random vars
        if not self.use_rand:
            self.rand_chans = 4
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
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.in_chans+self.rand_chans), fd, fd),
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
        self.w3 = weight_ifn((self.out_chans, (self.in_chans+self.rand_chans), fd, fd),
                                "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.g3, self.b3])
        # derp a derp parameterrrrr
        self.wx = weight_ifn((self.in_chans, self.rand_chans, 1, 1),
                             "{}_wx".format(self.mod_name))
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
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
        ss = self.us_stride            # stride for "learned upsampling"
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
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=0.01,
                                          dtype=theano.config.floatX)
        else:
            if not self.use_rand:
                # mask out random values, so they won't get used
                rand_vals = 0.0 * rand_vals
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape # return vals must be theano vars

        # apply dropout to input
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)

        if self.mod_type == 1:
            # perturb top-down activations based on rand_vals
            act_pert = dnn_conv(rand_vals, self.wx, subsample=(1, 1), border_mode=(0, 0))
            input = self.act_func( input + act_pert )
            # stack random values on top of input
            full_input = T.concatenate([0.0*rand_vals, input], axis=1)
        else:
            # stack random values on top of input
            full_input = T.concatenate([rand_vals, input], axis=1)

        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
            # apply direct input->output "projection" layer
            h3 = deconv(full_input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct input->output "projection" layer
            h4 = deconv(full_input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
        if self.apply_bn:
            h4 = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h4 = h4 + self.b3.dimshuffle('x',0,'x','x')
            h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)

        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
        return result

###########################################
# GENERATOR DOUBLE FULLY-CONNECTED MODULE #
###########################################

class GenFCResModule(object):
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
                 use_rand=True, use_fc=True,
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu',
                 mod_name='gm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.rand_chans = rand_chans
        self.fc_chans = fc_chans
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
        # use small dummy rand size if we won't use random vars
        if not self.use_rand:
            self.rand_chans = 4
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
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
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

        # TEMP TEMP TEMP
        input = self.act_func( input + T.dot(rand_vals, self.wx) )

        # stack random values on top of input
        full_input = T.concatenate([0.0*rand_vals, input], axis=1)

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
            # apply direct input->output layer
            h3 = T.dot(full_input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # only apply direct input->output layer
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
        act_func: ---
        unif_drop: drop rate for uniform dropout
        chan_drop: drop rate for channel-wise dropout
        apply_bn: whether to apply batch normalization
        use_td_cond: whether to use top-down conditioning
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self,
                 td_chans, bu_chans, rand_chans, conv_chans,
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
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
        self.use_bn_params = use_bn_params
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
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters
        self.w1_im = weight_ifn((self.conv_chans, (self.td_chans+self.bu_chans), 3, 3),
                                "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.conv_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.params.extend([self.w2_im])
        # initialize convolutional projection layer parameters
        self.w3_im = weight_ifn((2*self.rand_chans, (self.td_chans+self.bu_chans), 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        #############################################
        # Initialize "generative" model parameters. #
        #############################################
        if self.use_td_cond:
            # initialize first conv layer parameters
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                     "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.params.extend([self.w2_td])
            # initialize convolutional projection layer parameters
            self.w3_td = weight_ifn((2*self.rand_chans, self.td_chans, 3, 3),
                                    "{}_w3_td".format(self.mod_name))
            self.b3_td = bias_ifn((2*self.rand_chans), "{}_b3_td".format(self.mod_name))
            self.params.extend([self.w3_td, self.b3_td])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        # load top-down conditioning parameters
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.w3_td.set_value(floatX(param_dict['w3_td']))
            self.b3_td.set_value(floatX(param_dict['b3_td']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        # dump top-down parameters
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['w3_td'] = self.w3_td.get_value(borrow=False)
            param_dict['b3_td'] = self.b3_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            if self.use_conv:
                # apply first internal conv layer
                h1 = dnn_conv(td_input, self.w1_td, subsample=(1, 1), border_mode=(1, 1))
                if self.apply_bn:
                    h1 = switchy_bn(h1, g=self.g1_td, b=self.b1_td, n=noise,
                                    use_gb=self.use_bn_params)
                else:
                    h1 = h1 + self.b1_td.dimshuffle('x',0,'x','x')
                    h1 = add_noise(h1, noise=noise)
                h1 = self.act_func(h1)
                # apply second internal conv layer
                h2 = dnn_conv(h1, self.w2_td, subsample=(1, 1), border_mode=(1, 1))
                # apply direct input->output conv layer
                h3 = dnn_conv(td_input, self.w3_td, subsample=(1, 1), border_mode=(1, 1))
                # combine non-linear and linear transforms of input...
                h4 = h2 + h3 + self.b3_td.dimshuffle('x',0,'x','x')
            else:
                # apply direct input->output conv layer
                h3 = dnn_conv(td_input, self.w3_td, subsample=(1, 1), border_mode=(1, 1))
                h4 = h3 + self.b3_td.dimshuffle('x',0,'x','x')
            # split output into "mean" and "log variance" components, for using in
            # Gaussian reparametrization.
            out_mean = h4[:,:self.rand_chans,:,:]
            out_logvar = h4[:,self.rand_chans:,:,:]
        else:
            # if no top-down conditioning, return ZMUV Gaussian params
            out_mean = 0.0
            out_logvar = 0.0
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, share_mask=False, noise=None):
        """
        Combine td_input and bu_input, to put distributions over some stuff.
        """
        # stack top-down and bottom-up inputs on top of each other
        full_input = T.concatenate([td_input, bu_input], axis=1)
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
                h1 = h1 + self.b1_im.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = dnn_conv(h1, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
            # apply direct input->output conv layer
            h3 = dnn_conv(full_input, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3 + self.b3_im.dimshuffle('x',0,'x','x')
        else:
            # apply direct input->output conv layer
            h3 = dnn_conv(full_input, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
            h4 = h3 + self.b3_im.dimshuffle('x',0,'x','x')

        # split output into "mean" and "log variance" components, for using in
        # Gaussian reparametrization.
        out_mean = h4[:,:self.rand_chans,:,:]
        out_logvar = h4[:,self.rand_chans:,:,:]
        return out_mean, out_logvar

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
        act_func: ---
        unif_drop: drop rate for uniform dropout
        apply_bn: whether to use batch normalization
        use_td_cond: whether to use top-down conditioning
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, td_chans, bu_chans, fc_chans, rand_chans,
                 use_fc=True, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_name='im_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.fc_chans = fc_chans
        self.rand_chans = rand_chans
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
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
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
        # initialize weights for transform into fc layer
        self.w1 = weight_ifn(((self.td_chans+self.bu_chans), self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        # initialize weights for transform out of fc layer
        self.w2 = weight_ifn((self.fc_chans, 2*self.rand_chans),
                             "{}_w2".format(self.mod_name))
        self.params.extend([self.w2])
        # initialize weights for transform straight from input to output
        self.w3 = weight_ifn(((self.td_chans+self.bu_chans), 2*self.rand_chans),
                    "{}_w3".format(self.mod_name))
        self.b3 = bias_ifn((2*self.rand_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Apply this fully connected inference module to the given input. This
        produces a set of means and log variances for some Gaussian variables.
        """
        batch_size = td_input.shape[0]
        rand_shape = (batch_size, self.rand_chans)
        # NOTE: top-down conditioning path is not implemented yet
        out_mean = cu_rng.norma1(size=rand_shape, avg=0.0, std=0.01,
                                 dtype=theano.config.floatX)
        out_logvar = cu_rng.norma1(size=rand_shape, avg=0.0, std=0.01,
                                   dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input, share_mask=False, noise=None):
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
            h1 = T.dot(full_input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward to from fc layer to output
            h2 = T.dot(h1, self.w2)
            # feedforward directly from bu_input to output
            h3 = T.dot(full_input, self.w3)
            h4 = h2 + h3 + self.b3.dimshuffle('x',0)
        else:
            # feedforward directly from bu_input to output
            h3 = T.dot(full_input, self.w3)
            h4 = h3 + self.b3.dimshuffle('x',0)
        # split output into mean and log variance parts
        out_mean = h4[:,:self.rand_chans]
        out_logvar = h4[:,self.rand_chans:]
        return out_mean, out_logvar

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
        act_func: ---
        unif_drop: drop rate for unifor dropout
        apply_bn: whether to use batchnormalization
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, bu_chans, fc_chans, rand_chans,
                 use_fc=True, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
                 use_bn_params=True,
                 mod_name='dm_fc'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.bu_chans = bu_chans
        self.fc_chans = fc_chans
        self.rand_chans = rand_chans
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
        self.use_bn_params = True
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
        # initialize weights for transform into fc layer
        self.w1 = weight_ifn((self.bu_chans, self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.g1, self.b1]
        # initialize weights for transform out of fc layer
        self.w2 = weight_ifn((self.fc_chans, 2*self.rand_chans),
                             "{}_w2".format(self.mod_name))
        self.params.extend([self.w2])
        # initialize weights for transform straight from input to output
        self.w3 = weight_ifn((self.bu_chans, 2*self.rand_chans),
                                "{}_w3".format(self.mod_name))
        self.b3 = bias_ifn((2*self.rand_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
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
        Dump model params directly to a dict of numpy arrays.
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
                h1 = h1 + self.b1.dimshuffle('x',0)
                h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward to from fc layer to output
            h2 = T.dot(h1, self.w2)
            # feedforward directly from bu_input to output
            h3 = T.dot(bu_input, self.w3)
            h4 = h2 + self.b3.dimshuffle('x',0) + h3
        else:
            # feedforward directly from bu_input to output
            h3 = T.dot(bu_input, self.w3)
            h4 = h3 + self.b3.dimshuffle('x',0)
        # split output into mean and log variance parts
        out_mean = h4[:,:self.rand_chans]
        out_logvar = h4[:,self.rand_chans:]
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
        self.w1 = weight_ifn((self.in_dim, self.out_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.out_dim), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_dim), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['g1'] = self.g1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input, share_mask=False, noise=None):
        """
        Apply this gfully connected module.
        """
        # flatten input to 1d per example
        h1 = T.flatten(hq, 2)
        # apply dropout
        h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
        # feed-forward through layer
        h2 = T.dot(h1, self.w1)
        if self.apply_bn:
            h3 = switchy_bn(h2, g=self.g1, b=self.b1, n=noise,
                            use_gb=self.use_bn_params)
        else:
            h3 = h2 + self.b1.dimshuffle('x',0)
            h3 = add_noise(h3, noise=noise)
        h4 = self.act_func(h3)
        return h4






##############
# EYE BUFFER #
##############
