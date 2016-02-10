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

######################################################################
# DOUBLE CONVOLUTIONAL RESIDUAL MODULE -- USED FOR TD AND BU MODULES #
######################################################################

class TdBuConvResModule(object):
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
        self.w3 = weight_ifn((self.out_chans, self.in_chans, fd, fd),
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

    def apply(self, input, share_mask=False, noise=None):
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
            # use the internal convolution layer
            if self.stride in ['double', 'single']:
                # apply first internal conv layer (we might downsample)
                h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
                if self.apply_bn:
                    h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                    use_gb=self.use_bn_params)
                else:
                    h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer
                h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
                # apply pass-through conv layer (might downsample)
                h3 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                # apply first internal conv layer (we're going to upsample)
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
                if self.apply_bn:
                    h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                    use_gb=self.use_bn_params)
                else:
                    h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer (might upsample)
                h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
                # apply pass-through conv layer (might upsample)
                h3 = deconv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of module input...
            h4 = h2 + h3
        else:
            # don't use the internal convolution layer
            if self.stride in ['double', 'single']:
                h4 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                h4 = deconv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
        if self.apply_bn:
            pre_act = switchy_bn(h4, g=self.g3, b=self.b3, n=noise,
                                 use_gb=self.use_bn_params)
        else:
            pre_act = h4 + self.b3.dimshuffle('x',0,'x','x')
        act = self.act_func(h4)
        return act, pre_act


####################################
# GENERATOR FULLY CONNECTED MODULE #
####################################

class TdBuFCResModule(object):
    """
    Module fully-connected residual-type transform.
    """
    def __init__(self,
                 in_dim, fc_dim, out_shape,
                 use_fc=True,
                 apply_bn=True,
                 unif_drop=0.0,
                 act_func='relu',
                 use_bn_params=True,
                 mod_name='dm_fc'):
        assert (len(out_shape) == 3) or (len(out_shape) == 1), \
                "out_shape should describe the input to conv or fc layer."
        self.in_dim = in_dim
        self.fc_dim = fc_dim
        self.out_shape = out_shape
        if len(out_shape) == 1:
            # output is going to a fully-connected layer
            self.out_dim = out_shape[0]
        else:
            # output is goind to a conv layer
            self.out_dim = out_shape[0] * out_shape[1] * out_shape[2]
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
        self.w1 = weight_ifn((self.in_dim, self.fc_dim),
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
        self.w3 = weight_ifn((self.in_dim, self.out_dim),
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

    def apply(self, input, share_mask=False, noise=None):
        """
        Apply this module. Return activations and pre-activations.
        """
        input = fc_drop_func(input, self.unif_drop, share_mask=share_mask)
        if self.use_fc:
            # use internal fully-connected layer
            h1 = T.dot(input, self.w1)
            if self.apply_bn:
                h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                                use_gb=self.use_bn_params)
            else:
                h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            h2 = T.dot(h1, self.w2) + T.dot(input, self.w3)
        else:
            # don't use internal fully-connected layer
            h2 = T.dot(input, self.w3)
        if self.apply_bn:
            pre_act = switchy_bn(h2, g=self.g3, b=self.b3, n=noise,
                                 use_gb=self.use_bn_params)
        else:
            pre_act = h2 + self.b3.dimshuffle('x',0)
        if len(self.out_shape) == 3:
            # output is for a convolutional layer, so we need to reshape it
            pre_act = h3.reshape((h3.shape[0], self.out_shape[0], \
                                  self.out_shape[1], self.out_shape[2]))
        act = self.act_func(pre_act)
        return result





#########################################
# GENERATOR DOUBLE CONVOLUTIONAL MODULE #
#########################################

class IMConvResModule(object):
    """
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        conv_chans: number of channels in the "internal" convolution layer
        cond_layers: hidden layers in conditioning transform (1 or 2)
        pert_layers: hidden layers in the perturbation transform (1 or 2)
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
                 cond_layers=1, pert_layers=1,
                 unif_drop=0.0, chan_drop=0.0,
                 act_func='relu',
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_name='gm_conv'):
        assert (cond_layers in [1, 2]), \
                "cond_layers must be 1 or 2."
        assert (pert_layers in [1, 2]), \
                "pert_layers must be 1 or 2."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
        self.cond_layers = cond_layers
        self.pert_layers = pert_layers
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
        #######################################################
        # Initialize parameters for the IM conditioning path. #
        #######################################################
        # initialize first hidden layer parameters
        self.w1_im = weight_ifn((self.conv_chans, (self.td_chans+self.bu_chans), 3, 3),
                                "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.conv_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize second hidden layer parameters
        self.w2_im = weight_ifn((self.conv_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.conv_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.conv_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        ####################################################
        # Initialize parameters for the perturbation path. #
        ####################################################
        # initialize first hidden layer parameters
        self.w1_pt = weight_ifn((self.conv_chans, self.rand_chans, 3, 3),
                                "{}_w1_pt".format(self.mod_name))
        self.g1_pt = gain_ifn((self.conv_chans), "{}_g1_pt".format(self.mod_name))
        self.b1_pt = bias_ifn((self.conv_chans), "{}_b1_pt".format(self.mod_name))
        self.params.extend([self.w1_pt, self.g1_pt, self.b1_pt])
        # initialize second hidden layer parameters
        self.w2_pt = weight_ifn((self.conv_chans, self.conv_chans, 3, 3),
                                "{}_w2_pt".format(self.mod_name))
        self.g2_pt = gain_ifn((self.conv_chans), "{}_g2_pt".format(self.mod_name))
        self.b2_pt = bias_ifn((self.conv_chans), "{}_b2_pt".format(self.mod_name))
        self.params.extend([self.w2_pt, self.g2_pt, self.b2_pt])
        # initialize perturbation layer parameters
        self.w3_pt = weight_ifn((self.td_chans, self.conv_chans, 3, 3),
                                "{}_w3_pt".format(self.mod_name))
        self.b3_pt = bias_ifn(self.td_chans, "{}_b3_pt".format(self.mod_name))
        self.params.extend([self.w3_pt, self.b3_pt])
        #######################################################
        # Initialize parameters for the TD conditioning path. #
        #######################################################
        if self.use_td_cond:
            # initialize first hidden layer parameters
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                     "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.g1_td, self.b1_td])
            # initialize second hidden layer parameters
            self.w2_td = weight_ifn((self.conv_chans, self.conv_chans, 3, 3),
                                     "{}_w2_td".format(self.mod_name))
            self.g2_td = gain_ifn((self.conv_chans), "{}_g2_td".format(self.mod_name))
            self.b2_td = bias_ifn((self.conv_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.g2_td, self.b2_td])
            # initialize conditioning layer parameters
            self.w3_td = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w3_td".format(self.mod_name))
            self.b3_td = bias_ifn((2*self.rand_chans), "{}_b3_td".format(self.mod_name))
            self.params.extend([self.w3_td, self.b3_td])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        # load IM conditioning path parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.g1_im.set_value(floatX(param_dict['g1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.g2_im.set_value(floatX(param_dict['g2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        # load perturbation path parameters
        self.w1_pt.set_value(floatX(param_dict['w1_pt']))
        self.g1_pt.set_value(floatX(param_dict['g1_pt']))
        self.b1_pt.set_value(floatX(param_dict['b1_pt']))
        self.w2_pt.set_value(floatX(param_dict['w2_pt']))
        self.g2_pt.set_value(floatX(param_dict['g2_pt']))
        self.b2_pt.set_value(floatX(param_dict['b2_pt']))
        self.w3_pt.set_value(floatX(param_dict['w3_pt']))
        self.b3_pt.set_value(floatX(param_dict['b3_pt']))
        # load TD conditioning path parameters
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.g1_td.set_value(floatX(param_dict['g1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.g2_td.set_value(floatX(param_dict['g2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
            self.w3_td.set_value(floatX(param_dict['w3_td']))
            self.b3_td.set_value(floatX(param_dict['b3_td']))
        return

    def dump_params(self):
        """
        Dump model params directly to a dict of numpy arrays.
        """
        param_dict = {}
        # dump IM conditioning path parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['g1_im'] = self.g1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['g2_im'] = self.g2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        # dump perturbation path parameters
        param_dict['w1_pt'] = self.w1_pt.get_value(borrow=False)
        param_dict['g1_pt'] = self.g1_pt.get_value(borrow=False)
        param_dict['b1_pt'] = self.b1_pt.get_value(borrow=False)
        param_dict['w2_pt'] = self.w2_pt.get_value(borrow=False)
        param_dict['g2_pt'] = self.g2_pt.get_value(borrow=False)
        param_dict['b2_pt'] = self.b2_pt.get_value(borrow=False)
        param_dict['w3_pt'] = self.w3_pt.get_value(borrow=False)
        param_dict['b3_pt'] = self.b3_pt.get_value(borrow=False)
        # dump TD conditioning path parameters
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['g1_td'] = self.g1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['g2_td'] = self.g2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
            param_dict['w3_td'] = self.w3_td.get_value(borrow=False)
            param_dict['b3_td'] = self.b3_td.get_value(borrow=False)
        return param_dict

    def apply_conv_layer_1(self, h, w, g, b, noise=None, share_mask=False):
        h = dnn_conv(h, w, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            h = switchy_bn(h, g=g, b=b, n=noise,
                           use_gb=self.use_bn_params)
        else:
            h = h + b.dimshuffle('x',0,'x','x')
        h = self.act_func(h)
        h = conv_drop_func(h, self.unif_drop, self.chan_drop,
                           share_mask=share_mask)

    def apply_td(self, td_pre_act, rand_vals=None, noise=None, share_mask=False):
        """
        Apply a stochastic perturbation to td_pre_act. Values for the latent
        variables controlling the perturbation can be given in rand_vals.
        """
        # NOTE: top-down conditioning isn't implemented yet...
        if rand_vals is None:
            # get a set of ZMUV Gauss samples if one wasn't given
            tdpas = td_pre_act.shape
            rand_shape = (tdpas[0], self.rand_chans, tdpas[2], tdpas[3])
            rand_vals = cu_rng.normal(size=rand_shape)
        # transform through first hidden layer
        h1_pt = dnn_conv(rand_vals, self.w1_pt, subsample=(1, 1), border_mode=(1, 1))
        if self.apply_bn:
            h1_pt = switchy_bn(h1_pt, g=self.g1_pt, b=self.b1_pt, n=noise,
                               use_gb=self.use_bn_params)
        else:
            h1_pt = h1_pt + self.b1_pt.dimshuffle('x',0,'x','x')
        h1_pt = self.act_func(h1_pt)
        h2_pt = conv_drop_func(h1_pt, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)
        # transform through second hidden layer, if desired
        if self.pert_layers == 2:
            h2_pt = dnn_conv(h2_pt, self.w2_pt, subsample=(1, 1), border_mode=(1, 1))
            if self.apply_bn:
                h2_pt = switchy_bn(h2_pt, g=self.g2_pt, b=self.b2_pt, n=noise,
                                   use_gb=self.use_bn_params)
            else:
                h2_pt = h2_pt + self.b2_pt.dimshuffle('x',0,'x','x')
            h1_pt = self.act_func(h1_pt)
            h2_pt = conv_drop_func(h1_pt, self.unif_drop, self.chan_drop,
                                   share_mask=share_mask)
        return td_act

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

class IMFCModule(object):
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
        self.w_out = weight_ifn((self.bu_chans, 2*self.rand_chans),
                                "{}_w_out".format(self.mod_name))
        self.b_out = bias_ifn((2*self.rand_chans), "{}_b_out".format(self.mod_name))
        self.params.extend([self.w_out, self.b_out])
        return

    def load_params(self, param_dict):
        """
        Load model params directly from a dict of numpy arrays.
        """
        self.w1.set_value(floatX(param_dict['w1']))
        self.g1.set_value(floatX(param_dict['g1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.w_out.set_value(floatX(param_dict['w_out']))
        self.b_out.set_value(floatX(param_dict['b_out']))
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
        param_dict['w_out'] = self.w_out.get_value(borrow=False)
        param_dict['b_out'] = self.b_out.get_value(borrow=False)
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
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward to from fc layer to output
            h2 = T.dot(h1, self.w2)
            # feedforward directly from bu_input to output
            h3 = T.dot(bu_input, self.w_out)
            h4 = h2 + self.b_out.dimshuffle('x',0) # + h3
        else:
            # feedforward directly from bu_input to output
            h3 = T.dot(bu_input, self.w_out)
            h4 = h3 + self.b_out.dimshuffle('x',0)
        # split output into mean and log variance parts
        out_mean = h4[:,:self.rand_chans]
        out_logvar = h4[:,self.rand_chans:]
        return out_mean, out_logvar








##############
# EYE BUFFER #
##############
