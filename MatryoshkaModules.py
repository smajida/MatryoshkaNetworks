import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
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
    """
    Helper function for optionally applying secondary shift+scale in BN.
    """
    if use_gb and (not (g is None) or (b is None)):
        bn_acts = batchnorm(acts, g=g, b=b, n=n)
    else:
        bn_acts = batchnorm(acts, n=n)
    return bn_acts


def wn_conv_op(input, w, g, b, stride='single', noise=None, bm=1):
    # set each output channel to have unit norm afferent weights
    # conv weight shape: (out_chans, in_chans, rows, cols)

    #w_norms = T.sqrt(T.sum(T.sum(T.sum(w**2.0, axis=3), axis=2), axis=1))
    #w_n = w / w_norms.dimshuffle(0,'x','x','x')

    w_n = w

    # compute convolution
    if stride == 'single':
        # no resizing
        h_pre = dnn_conv(input, w_n, subsample=(1, 1), border_mode=(bm, bm))
    elif stride == 'double':
        # downsampling, 2x2 stride
        h_pre = dnn_conv(input, w_n, subsample=(2, 2), border_mode=(bm, bm))
    else:
        # upsampling, 0.5x0.5 stride
        h_pre = deconv(input, w_n, subsample=(2, 2), border_mode=(bm, bm))
    # compute channel-wise stats before rescale and shift
    # conv result shape: (batch, out_chans, rows, cols)
    pre_mean = T.mean(T.mean(T.mean(h_pre, axis=0), axis=2), axis=1)
    pre_res = (h_pre - pre_mean.dimshuffle('x',0,'x','x'))
    pre_std = T.sqrt(T.mean(T.mean(T.mean(pre_res**2.0, axis=0), axis=2), axis=1))

    # rescale and shift
    h_post = h_pre + b.dimshuffle('x',0,'x','x') # channel-wise shift
    #h_post = h_pre * g.dimshuffle('x',0,'x','x')  # channel-wise rescale
    #h_post = h_post + b.dimshuffle('x',0,'x','x') # channel-wise shift

    # compute channel-wise stats after rescale and shift
    post_mean = T.mean(T.mean(T.mean(h_post, axis=0), axis=2), axis=1)
    post_res = (h_post - post_mean.dimshuffle('x',0,'x','x'))
    post_std = T.sqrt(T.mean(T.mean(T.mean(post_res**2.0, axis=0), axis=2), axis=1))

    # add noise
    h_post = add_noise(h_post, noise=noise)

    # return important quantities in a dict
    res_dict = {'h_pre': h_pre, 'pre_mean': pre_mean, 'pre_std': pre_std,
                'h_post': h_post, 'post_mean': post_mean, 'post_std': post_std}
    return res_dict

def wn_fc_op(input, w, g, b, noise=None):
    # set each output channel to have unit norm afferent weights
    # fc weight shape: (in_chans, out_chans)

    #w_norms = T.sqrt(T.sum(w**2.0, axis=0))
    #w_n = w / w_norms.dimshuffle('x',0)

    w_n = w

    # compute initial linear transform
    h_pre = T.dot(input, w_n)
    # compute channel-wise stats before rescale and shift
    # conv result shape: (batch, out_chans, rows, cols)
    pre_mean = T.mean(h_pre, axis=0)
    pre_res = (h_pre - pre_mean.dimshuffle('x',0))
    pre_std = T.sqrt(T.mean(pre_res**2.0, axis=0))

    # rescale and shift
    h_post = h_pre + b.dimshuffle('x',0) # channel-wise shift
    #h_post = h_pre * g.dimshuffle('x',0)  # channel-wise rescale
    #h_post = h_post + b.dimshuffle('x',0) # channel-wise shift

    # compute channel-wise stats after rescale and shift
    post_mean = T.mean(h_post, axis=0)
    post_res = (h_post - post_mean.dimshuffle('x',0))
    post_std = T.sqrt(T.mean(post_res**2.0, axis=0))

    # add noise
    h_post = add_noise(h_post, noise=noise)

    # return important quantities in a dict
    res_dict = {'h_pre': h_pre, 'pre_mean': pre_mean, 'pre_std': pre_std,
                'h_post': h_post, 'post_mean': post_mean, 'post_std': post_std}
    return res_dict

def wn_costs(res_dict):
    """
    Compute mean and standard deviation "normalization costs", to be used
    for optimization-based initialization with weight normalization.
    """
    post_mean = res_dict['post_mean']
    post_std = res_dict['post_std']
    # make costs to encourage zero mean and unit standard deviation
    mean_cost = T.mean(post_mean**2.0)
    std_cost = T.mean((post_std - 0.1)**2.0)
    return mean_cost, std_cost

#######################################
# BASIC DOUBLE FULLY-CONNECTED MODULE #
#######################################

class BasicFCResModule(object):
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
            h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # apply second internal linear layer
            h2 = T.dot(h1, self.w2)
            # apply short-cut linear layer
            h3 = T.dot(input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply short-cut linear layer
            h4 = T.dot(input, self.w3)
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        # linaer transform followed by activations and stuff
        h1 = T.dot(input, self.w1)
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
    Module with a direct short-cut connection that gets with the output of a
    conv->activation->conv transformation.

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
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
                h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer
                h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
                # apply short-cut conv layer (might downsample)
                h3 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                # apply first internal conv layer
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
                h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
                h1 = add_noise(h1, noise=noise)
                h1 = self.act_func(h1)
                # apply dropout at intermediate convolution layer
                h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
                # apply second internal conv layer (might upsample)
                h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
                # apply short-cut conv layer (might upsample)
                h3 = deconv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct short-cut conv layer
            if self.stride in ['double', 'single']:
                h4 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            else:
                h4 = deconv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
        h4 = h4 + self.b3.dimshuffle('x',0,'x','x')
        h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
        if rand_shapes:
            result = [output, input.shape]
        else:
            result = output
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
        assert (filt_shape == (3,3) or filt_shape == (5,5)), \
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

        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, self.in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize alternate conv layer parameters
        self.w3 = weight_ifn((self.out_chans, self.in_chans, fd, fd),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        # gain and bias parameters are involved in weight normalization
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
        self.params.extend([self.w1, self.b1])
        # initialize second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # initialize alternate conv layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.b3])
        # gain and bias parameters are involved in weight normalization
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
        batch_size = input.shape[0] # number of inputs in this batch
        bm = (self.filt_dim - 1) // 2
        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)

        # apply first internal conv layer
        h1_dict = wn_conv_op(input, w=self.w1, g=self.g1, b=self.b1,
                             stride='single', noise=noise, bm=bm)
        h1 = self.act_func(h1_dict['h_post'])
        h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                            share_mask=share_mask)

        # apply second internal conv layer
        h2_dict = wn_conv_op(h1, w=self.w2, g=self.g2, b=self.b2,
                             stride='single', noise=noise, bm=bm)
        h2 = h2_dict['h_post']

        # apply perturbation and non-linearity to input
        h3 = input + h2
        h3 = h3 * self.g3.dimshuffle('x',0,'x','x')
        h3 = h3 + self.b3.dimshuffle('x',0,'x','x')
        h3 = add_noise(h3, noise=noise)
        h3 = self.act_func(h3)

        # compute costs for optimization-based initialization
        h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
        h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
        self.wn_mean_cost = h1_mean_cost + h2_mean_cost
        self.wn_std_cost = h1_std_cost + h2_std_cost

        if rand_shapes:
            result = [h3, input.shape]
        else:
            result = h3
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
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.out_chans, self.in_chans, self.filt_dim, self.filt_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.out_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.b1]
        # bunch up the weight normalization parameters
        self.wn_params = [self.g1, self.b1]
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
        bm = int((self.filt_dim - 1) / 2) # use "same" mode convolutions
        # apply uniform and/or channel-wise dropout if desired
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)

        h1_dict = wn_conv_op(input, w=self.w1, g=self.g1, b=self.b1,
                             stride=self.stride, noise=noise, bm=bm)
        h1 = self.act_func(h1_dict['h_post'])

        # compute costs for optimization-based initialization
        h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
        self.wn_mean_cost = h1_mean_cost
        self.wn_std_cost = h1_std_cost

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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        batch_size = input.shape[0] # number of inputs in this batch
        ss = self.ds_stride         # stride for "learned downsampling"
        bm = (self.filt_dim - 1) // 2 # set border mode for the convolutions
        # apply dropout to input
        input = conv_drop_func(input, self.unif_drop, self.chan_drop,
                               share_mask=share_mask)
        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(input, self.w1, subsample=(ss, ss), border_mode=(bm, bm))
            h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
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
            h4 = h4 + self.b3.dimshuffle('x',0,'x','x')
            h4 = add_noise(h4, noise=noise)
            output = self.act_func(h4)
        else:
            # apply direct input->output short-cut layer
            h3 = dnn_conv(input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            h3 = h3 + self.b3.dimshuffle('x',0,'x','x')
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
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        # initialize first layer parameters
        self.w1 = weight_ifn((self.rand_dim, self.fc_dim),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_dim), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_dim), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize second layer parameters
        self.w2 = weight_ifn((self.fc_dim, self.out_dim),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.out_dim), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_dim), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize single layer parameters
        self.w3 = weight_ifn((self.rand_dim, self.out_dim),
                             "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((self.out_dim), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((self.out_dim), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        # record weight normalization parameters
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
            rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=1.0, \
                                      dtype=theano.config.floatX)
        else:
            # get the shape of the incoming latent variables
            rand_shape = (rand_vals.shape[0], self.rand_dim)
        rand_vals = rand_vals.reshape(rand_shape)
        rand_shape = rand_vals.shape

        if self.use_fc:
            # apply first internal fc layer
            h1_dict = wn_fc_op(rand_vals, w=self.w1, g=self.g1, b=self.b1,
                               noise=noise)
            h1 = self.act_func(h1_dict['h_post'])
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)

            # feedforward from fc layer to output
            h2_dict = wn_fc_op(h1, w=self.w2, g=self.g2, b=self.b2,
                               noise=noise)
            h2 = self.act_func(h2_dict['h_post'])

            # compute costs for optimization-based initialization
            h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h1_mean_cost + h2_mean_cost
            self.wn_std_cost = h1_std_cost + h2_std_cost

        else:
            # feedforward directly from rand_vals
            h2_dict = wn_fc_op(rand_vals, w=self.w3, g=self.g3, b=self.b3,
                               noise=noise)
            h2 = self.act_func(h2_dict['h_post'])

            # compute costs for optimization-based initialization
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h2_mean_cost
            self.wn_std_cost = h2_std_cost

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
    Module of one regular convolution layer followed by one "fractionally-
    strided" convolution layer, which gets combined with the output of a
    "fractionally-strided" short-cut layer. Inputs to this module will get
    combined with some latent variables prior to processing.

    Params:
        in_chans: number of channels in the inputs to module
        out_chans: number of channels in the outputs from module
        conv_chans: number of channels in the "internal" convolution layer
        rand_chans: number of random channels to augment input
        filt_shape: size of filters (either (3, 3) or (5, 5))
        use_rand: flag for whether or not to augment inputs
        use_conv: flag for whether to use "internal" convolution layer
        us_stride: upsampling ratio in the fractionally-strided convolution
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
                 use_bn_params=True, act_func='relu', mod_type=0,
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        self.wx = weight_ifn((self.conv_chans, self.rand_chans, 3, 3),
                             "{}_wx".format(self.mod_name))
        self.wy = weight_ifn((self.in_chans, self.conv_chans, 3, 3),
                             "{}_wy".format(self.mod_name))
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
                rand_vals = cu_rng.normal(size=rand_shape, avg=0.0, std=0.001,
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

        # stack random values on top of input
        full_input = T.concatenate([rand_vals, input], axis=1)

        if self.use_conv:
            # apply first internal conv layer
            h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
            h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
            h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2 = deconv(h1, self.w2, subsample=(ss, ss), border_mode=(bm, bm))
            # apply direct input->output short-cut layer
            h3 = deconv(full_input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # apply direct input->output short-cut layer
            h4 = deconv(full_input, self.w3, subsample=(ss, ss), border_mode=(bm, bm))
        h4 = h4 + self.b3.dimshuffle('x',0,'x','x')
        h4 = add_noise(h4, noise=noise)
        output = self.act_func(h4)
        if rand_shapes:
            result = [output, rand_shape]
        else:
            result = output
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
                 use_rand=True, use_conv=True, us_stride=2,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu', mod_type=0,
                 mod_name='gm_conv'):
        assert ((us_stride == 1)), \
                "us_stride must be 1."
        assert ((in_chans == out_chans)), \
                "in_chans == out_chans is required."
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.in_chans+self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.conv_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.conv_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.conv_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.conv_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize third conv layer parameters
        self.w3 = weight_ifn((2*self.out_chans, self.conv_chans, fd, fd),
                                "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((2*self.out_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((2*self.out_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        # record weight normalization parameters
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
        self.params.extend([self.w1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.g2 = source_module.g2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # share third conv layer parameters
        self.w3 = source_module.w3
        self.g3 = source_module.g3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.b3])
        # record weight normalization parameters
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
        rand_shape = rand_vals.shape # return vals must be theano vars

        pert_input = T.concatenate([rand_vals, input], axis=1)
        # apply first internal conv layer
        h1_dict = wn_conv_op(pert_input, w=self.w1, g=self.g1, b=self.b1,
                             stride='single', noise=noise, bm=bm)
        h1 = self.act_func(h1_dict['h_post'])

        # # apply second internal conv layer
        # h2_dict = wn_conv_op(h1, w=self.w2, g=self.g2, b=self.b2,
        #                      stride='single', noise=noise, bm=bm)
        # h2 = self.act_func(h2_dict['h_post'])

        # compute perturbation and gating values
        h3_dict = wn_conv_op(h1, w=self.w3, g=self.g3, b=self.b3,
                             stride='single', noise=noise, bm=bm)
        h3 = h3_dict['h_post']
        h3_pert = h3[:,:self.out_chans,:,:]
        h3_gate = h3[:,self.out_chans:,:,:]

        # combine non-linear and linear transforms of input...
        h4 = (sigmoid(h3_gate + 1.0) * input) + h3_pert
        output = self.act_func(h4)

        # compute costs for optimization-based initialization
        h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
        #h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
        h3_mean_cost, h3_std_cost = wn_costs(h3_dict)
        self.wn_mean_cost = h1_mean_cost + h3_mean_cost #+ h2_mean_cost
        self.wn_std_cost = h1_std_cost + h3_std_cost #+ h2_std_cost

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
                 in_chans, out_chans, conv_chans, rand_chans, filt_shape,
                 use_rand=True, use_conv=True, us_stride=2,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu', mod_type=0,
                 mod_name='gm_conv'):
        assert ((us_stride == 1)), \
                "us_stride must be 1."
        assert ((in_chans == out_chans)), \
                "in_chans == out_chans is required."
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize gate layer parameters
        self.w1 = weight_ifn((2*self.in_chans, (self.in_chans+self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((2*self.in_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((2*self.in_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])
        # initialize first new state layer parameters
        self.w2 = weight_ifn((self.in_chans, (self.in_chans+self.rand_chans), fd, fd),
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
        rand_shape = rand_vals.shape # return vals must be theano vars

        # compute update gate and remember gate
        gate_input = T.concatenate([rand_vals, input], axis=1)
        h = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        h = h + self.b1.dimshuffle('x',0,'x','x')
        h = add_noise(h, noise=noise)
        h = sigmoid(h + 1.)
        u = h[:,:self.in_chans,:,:]
        r = h[:,self.in_chans:,:,:]
        # compute new state proposal -- include hidden layer
        state_input = T.concatenate([rand_vals, r*input], axis=1)
        s = dnn_conv(state_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        s = s + self.b2.dimshuffle('x',0,'x','x')
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        input = self.act_func( input + T.dot(rand_vals, self.wx) )

        # stack random values on top of input
        full_input = T.concatenate([0.0*rand_vals, input], axis=1)

        if self.use_fc:
            # apply first internal fc layer
            h1 = T.dot(full_input, self.w1)
            h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # apply second internal fc layer
            h2 = T.dot(h1, self.w2)
            # apply direct short-cut layer
            h3 = T.dot(full_input, self.w3)
            # combine non-linear and linear transforms of input...
            h4 = h2 + h3
        else:
            # only apply direct short-cut layer
            h4 = T.dot(full_input, self.w3)
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
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
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
        self.use_bn_params = True
        self.mod_type = mod_type
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        if self.mod_type == 0:
            self.w1_im = weight_ifn((2*self.im_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        else:
            self.w1_im = weight_ifn((2*self.im_chans, (3*self.td_chans+self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((2*self.im_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((2*self.im_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.g1_im, self.b1_im])
        # initialize GRU state update parameters
        if self.mod_type == 0:
            self.w2_im = weight_ifn((self.im_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                    "{}_w2_im".format(self.mod_name))
        else:
            self.w2_im = weight_ifn((self.im_chans, (3*self.td_chans+self.im_chans), 3, 3),
                                    "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.im_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.im_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.g2_im, self.b2_im])
        # initialize conditioning parameters
        self.w3_im = weight_ifn((2*self.rand_chans, self.im_chans, 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((2*self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.g3_im, self.b3_im])
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
        # initialize conditioning parameters
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
            T.alloc(0.0, b_size, self.im_chans, rows, cols)

        # prepare input to gating functions
        if self.mod_type == 0:
            gate_input = T.concatenate([td_input, bu_input, im_input], axis=1)
        else:
            gate_input = T.concatenate([td_input, bu_input, td_input-bu_input, im_input], axis=1)
        gate_input = conv_drop_func(gate_input, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
        # compute gating information for GRU state update
        h1 = dnn_conv(gate_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
        h1 = h1 + self.b1_im.dimshuffle('x',0,'x','x')
        h1 = add_noise(h1, noise=noise)
        h1 = sigmoid(h1 + 1.)
        u = h1[:,:self.im_chans,:,:]
        r = h1[:,self.im_chans:,:,:]

        # prepare input for computing new state
        if self.mod_type == 0:
            state_input = T.concatenate([td_input, bu_input, r*im_input], axis=1)
        else:
            state_input = T.concatenate([td_input, bu_input, td_input-bu_input, r*im_input], axis=1)
        state_input = conv_drop_func(state_input, self.unif_drop, self.chan_drop,
                                     share_mask=share_mask)
        # compute new state for GRU state update
        h2 = dnn_conv(state_input, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
        h2 = h2 + self.b2_im.dimshuffle('x',0,'x','x')
        h2 = h2 + add_noise(h2, noise=noise)
        # perform GRU-style state update (for IM state)
        out_im = (u * im_input) + ((1. - u) * self.act_func(h2))

        # compute conditioning parameters
        h3 = dnn_conv(out_im, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
        h3 = h3 + self.b3_im.dimshuffle('x',0,'x','x')
        h3 = h3 + add_noise(h3, noise=noise)
        out_mean = h3[:,:self.rand_chans,:,:]
        out_logvar = h3[:,self.rand_chans:,:,:]
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
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
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
        self.use_bn_params = True
        self.mod_type = mod_type
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters (from input -> hidden layer)
        if self.mod_type == 0:
            self.w1_im = weight_ifn((self.conv_chans, (self.td_chans+self.bu_chans+self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        else:
            self.w1_im = weight_ifn((self.conv_chans, (3*self.td_chans+self.im_chans), 3, 3),
                                    "{}_w1_im".format(self.mod_name))
        self.g1_im = gain_ifn((self.conv_chans), "{}_g1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.b1_im])
        # initialize second conv layer parameters (from hidden layer -> IM state perturbation)
        self.w2_im = weight_ifn((self.im_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((self.im_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.im_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # initialize convolutional projection layer parameters
        self.w3_im = weight_ifn((2*self.rand_chans, self.im_chans, 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.g3_im = gain_ifn((2*self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # record weight normalization parameters
        self.wn_params = [self.g1_im, self.b1_im, self.g2_im, self.b2_im,
                          self.g3_im, self.b3_im]
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.g2_td = gain_ifn((2*self.rand_chans), "{}_g2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2*self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
            # add TD conditioning weight normalization parameters
            self.wn_params.extend([self.g1_td, self.b1_td, self.g2_td, self.b2_td])
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
        self.params.extend([self.w1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.b3_im])
        # weight normalization params
        self.wn_params = [self.g1_im, self.b1_im, self.g2_im, self.b2_im,
                          self.g3_im, self.b3_im]
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = source_module.w2_td
            self.g2_td = source_module.g2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
            # add TD conditioning weight normalization parameters
            self.wn_params.extend([self.g1_td, self.b1_td, self.g2_td, self.b2_td])
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
            self.g2_td.set_value(floatX(param_dict['g2_td']))
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
            param_dict['g2_td'] = self.g2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            # transform into hidden layer
            h1_dict = wn_conv_op(td_input, w=self.w1_td, g=self.g1_td, b=self.b1_td,
                                 stride='single', noise=noise, bm=1)
            h1 = self.act_func(h1_dict['h_post'])
            # transform to conditioning parameters
            h2_dict = wn_conv_op(h1, w=self.w2_td, g=self.g2_td, b=self.b2_td,
                                 stride='single', noise=noise, bm=1)
            h2 = h2_dict['h_post']

            out_mean = h2[:,:self.rand_chans,:,:]
            out_logvar = h2[:,self.rand_chans:,:,:]
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
            full_input = T.concatenate([td_input, bu_input, td_input-bu_input, im_input], axis=1)
        # do dropout
        full_input = conv_drop_func(full_input, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)

        # apply first internal conv layer
        h1_dict = wn_conv_op(full_input, w=self.w1_im, g=self.g1_im, b=self.b1_im,
                             stride='single', noise=noise, bm=1)
        h1 = self.act_func(h1_dict['h_post'])
        h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                            share_mask=share_mask)

        # apply second internal conv layer
        h2_dict = wn_conv_op(h1, w=self.w2_im, g=self.g2_im, b=self.b2_im,
                             stride='single', noise=noise, bm=1)
        h2 = h2_dict['h_post']

        # apply perturbation to IM input, then apply non-linearity
        out_im = self.act_func(im_input + h2)

        # compute conditional parameters from the updated IM state
        h3_dict = wn_conv_op(out_im, w=self.w3_im, g=self.g3_im, b=self.b3_im,
                             stride='single', noise=noise, bm=1)
        h3 = h3_dict['h_post']
        out_mean = h3[:,:self.rand_chans,:,:]
        out_logvar = h3[:,self.rand_chans:,:,:]

        # compute costs for optimization-based initialization
        h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
        h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
        h3_mean_cost, h3_std_cost = wn_costs(h3_dict)
        self.wn_mean_cost = h1_mean_cost + h2_mean_cost + h3_mean_cost
        self.wn_std_cost = h1_std_cost + h2_std_cost + h3_std_cost
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
                 use_conv=True, act_func='relu',
                 unif_drop=0.0, chan_drop=0.0,
                 apply_bn=True,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_type=0,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
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
        self.use_bn_params = True
        self.mod_type = mod_type
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        self.params.extend([self.w1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.g2_im = gain_ifn((2*self.rand_chans), "{}_g2_im".format(self.mod_name))
        self.b2_im = bias_ifn((2*self.rand_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # initialize convolutional projection layer parameters
        if self.mod_type == 0:
            # module acts just on TD and BU input
            self.w3_im = weight_ifn((2*self.rand_chans, (self.td_chans+self.bu_chans), 3, 3),
                                    "{}_w3_im".format(self.mod_name))
        else:
            # module acts on TD and BU input, and their difference
            self.w3_im = weight_ifn((2*self.rand_chans, (3*self.td_chans), 3, 3),
                                    "{}_w3_im".format(self.mod_name))
        self.g3_im = bias_ifn((2*self.rand_chans), "{}_g3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2*self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # record weight normalization parameters
        self.wn_params = [self.g1_im, self.b1_im, self.g2_im, self.b2_im,
                          self.g3_im, self.b3_im]

        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.g1_td = gain_ifn((self.conv_chans), "{}_g1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2*self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.g2_td = gain_ifn((2*self.rand_chans), "{}_g2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2*self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
            # add TD conditioning weight normalization parameters
            self.wn_params.extend([self.g1_td, self.b1_td, self.g2_td, self.b2_td])
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
        self.params.extend([self.w1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = source_module.w2_im
        self.g2_im = source_module.g2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.b2_im])
        # initialize conditioning layer parameters
        self.w3_im = source_module.w3_im
        self.g3_im = source_module.g3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.b3_im])
        # weight normalization params
        self.wn_params = [self.g1_im, self.b1_im, self.g2_im, self.b2_im,
                          self.g3_im, self.b3_im]
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.g1_td = source_module.g1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = source_module.w2_td
            self.g2_td = source_module.g2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
            # add TD conditioning weight normalization parameters
            self.wn_params.extend([self.g1_td, self.b1_td, self.g2_td, self.b2_td])
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
            self.g2_td.set_value(floatX(param_dict['g2_td']))
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
            param_dict['g2_td'] = self.g2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input, noise=None):
        """
        Put distributions over stuff based on td_input.
        """
        if self.use_td_cond:
            # transform into hidden layer
            h1_dict = wn_conv_op(td_input, w=self.w1_td, g=self.g1_td, b=self.b1_td,
                                 stride='single', noise=noise, bm=1)
            h1 = self.act_func(h1_dict['h_post'])
            # transform to conditioning parameters
            h2_dict = wn_conv_op(h1, w=self.w2_td, g=self.g2_td, b=self.b2_td,
                                 stride='single', noise=noise, bm=1)
            h2 = h2_dict['h_post']

            out_mean = h2[:,:self.rand_chans,:,:]
            out_logvar = h2[:,self.rand_chans:,:,:]
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
            full_input = T.concatenate([td_input, bu_input, td_input-bu_input], axis=1)
        # do dropout
        full_input = conv_drop_func(full_input, self.unif_drop, self.chan_drop,
                                    share_mask=share_mask)
        if self.use_conv:
            # apply first internal conv layer
            h1_dict = wn_conv_op(full_input, w=self.w1_im, g=self.g1_im, b=self.b1_im,
                                 stride='single', noise=noise, bm=1)
            h1 = self.act_func(h1_dict['h_post'])
            h1 = conv_drop_func(h1, self.unif_drop, self.chan_drop,
                                share_mask=share_mask)
            # apply second internal conv layer
            h2_dict = wn_conv_op(h1, w=self.w2_im, g=self.g2_im, b=self.b2_im,
                                 stride='single', noise=noise, bm=1)
            h2 = h2_dict['h_post']

            # compute costs for optimization-based initialization
            h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h1_mean_cost + h2_mean_cost
            self.wn_std_cost = h1_std_cost + h2_std_cost
        else:
            # apply direct short-cut conv layer
            h2_dict = wn_conv_op(full_input, w=self.w3_im, g=self.g3_im, b=self.b3_im,
                                 stride='single', noise=noise, bm=1)
            h2 = h2_dict['h_post']
            # compute wn init costs
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h2_mean_cost
            self.wn_std_cost = h2_std_cost

        # split output into "mean" and "log variance" components, for using in
        # Gaussian reparametrization.
        out_mean = h2[:,:self.rand_chans,:,:]
        out_logvar = h2[:,self.rand_chans:,:,:]
        return out_mean, out_logvar, None

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
        use_bn_params: whether to use BN params
        mod_name: text name for identifying module in theano graph
    """
    def __init__(self, td_chans, bu_chans, fc_chans, rand_chans,
                 use_fc=True, act_func='relu',
                 unif_drop=0.0, apply_bn=True,
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
        self.use_bn_params = True
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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

    def apply_td(self, td_input, noise=None):
        """
        Apply this fully connected inference module to the given input. This
        produces a set of means and log variances for some Gaussian variables.
        """
        batch_size = td_input.shape[0]
        rand_shape = (batch_size, self.rand_chans)
        # NOTE: top-down conditioning path is not implemented yet
        out_mean = cu_rng.norma1(size=rand_shape, avg=0.0, std=0.001,
                                 dtype=theano.config.floatX)
        out_logvar = cu_rng.norma1(size=rand_shape, avg=0.0, std=0.001,
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
            h1 = h1 + self.b1.dimshuffle('x',0)
            h1 = add_noise(h1, noise=noise)
            h1 = self.act_func(h1)
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
            # feedforward from fc layer to output
            h2 = T.dot(h1, self.w2)
            # feedforward directly from BU/TD inputs to output
            h3 = T.dot(full_input, self.w3)
            h4 = h2 + h3 + self.b3.dimshuffle('x',0)
        else:
            # feedforward directly from BU input to output
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
        apply_bn: whether to use batch normalization
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
        Initialize parameters for the layers in this module.
        """
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
        bias_ifn = inits.Constant(c=0.)
        # initialize weights for transform into fc layer
        self.w1 = weight_ifn((self.bu_chans, self.fc_chans),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.fc_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.fc_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.b1]
        # initialize weights for transform out of fc layer
        self.w2 = weight_ifn((self.fc_chans, 2*self.rand_chans),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((2*self.rand_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((2*self.rand_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize weights for transform straight from input to output
        self.w3 = weight_ifn((self.bu_chans, 2*self.rand_chans),
                                "{}_w3".format(self.mod_name))
        self.g3 = gain_ifn((2*self.rand_chans), "{}_g3".format(self.mod_name))
        self.b3 = bias_ifn((2*self.rand_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        # gain and bias parameters are involved in weight normalization
        self.wn_params = [self.g1, self.b1, self.g2, self.b2,
                          self.g3, self.b3]
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
            # apply first internal fc layer
            h1_dict = wn_fc_op(bu_input, w=self.w1, g=self.g1, b=self.b1,
                               noise=noise)
            h1 = self.act_func(h1_dict['h_post'])
            h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)

            # feedforward from fc layer to output
            h2_dict = wn_fc_op(h1, w=self.w2, g=self.g2, b=self.b2,
                               noise=None)
            h2 = h2_dict['h_post']

            # compute costs for optimization-based initialization
            h1_mean_cost, h1_std_cost = wn_costs(h1_dict)
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h1_mean_cost + h2_mean_cost
            self.wn_std_cost = h1_std_cost + h2_std_cost

        else:
            # feedforward directly from bu_input
            h2_dict = wn_fc_op(bu_input, w=self.w3, g=self.g3, b=self.b3,
                               noise=None)
            h2 = h2_dict['h_post']

            # compute costs for optimization-based initialization
            h2_mean_cost, h2_std_cost = wn_costs(h2_dict)
            self.wn_mean_cost = h2_mean_cost
            self.wn_std_cost = h2_std_cost

        # split output into mean and log variance parts
        out_mean = h2[:,:self.rand_chans]
        out_logvar = h2[:,self.rand_chans:]
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
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.03)
        gain_ifn = inits.Normal(loc=1., scale=0.03)
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
        h1 = T.flatten(hq, 2)
        # apply dropout
        h1 = fc_drop_func(h1, self.unif_drop, share_mask=share_mask)
        # feed-forward through layer
        h2 = T.dot(h1, self.w1)
        h3 = h2 + self.b1.dimshuffle('x',0)
        h3 = add_noise(h3, noise=noise)
        h4 = self.act_func(h3)
        return h4






##############
# EYE BUFFER #
##############
