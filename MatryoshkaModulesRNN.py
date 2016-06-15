import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import inits
from lib.rng import cu_rng
from lib.theano_utils import floatX
from lib.ops import deconv
from MatryoshkaModules import *

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify(leak=0.1)
bce = T.nnet.binary_crossentropy
hard_sigmoid = T.nnet.hard_sigmoid
tanh = activations.Tanh()
elu = activations.ELU()


def clip_softmax(x, axis=2):
    x = T.clip(x, -10., 10.)
    x = x - T.max(x, axis=axis, keepdims=True)
    x = T.exp(x)
    x = x / T.sum(x, axis=axis, keepdims=True)
    return x


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -10.0, 10.0))
    return output


class UDSampler2x(object):
    '''
    Module for 2x upsampling or downsampling.

    -- upsampling is by "repeat" and downsampling is by max pooling.

    Params:
        uds_type: sampling type in ['upsample', 'downsample']
        is_1d: whether this module will operate on 1d sequences
        mod_name: text name to identify this module in theano graph
    '''
    def __init__(self,
                 uds_type,
                 is_1d=False,
                 mod_name='basic_conv'):
        assert (uds_type in ['upsample', 'downsample'])
        self.uds_type = uds_type
        self.is_1d = is_1d
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        # dummy param, don't know if modules work wiht no params
        bias_ifn = inits.Constant(c=0.)
        self.b = bias_ifn((2), "{}_b".format(self.mod_name))
        self.params = [self.b]
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.b.set_value(floatX(param_dict['b']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['b'] = self.b.get_value(borrow=False)
        return param_dict

    def apply(self, input):
        '''
        Apply this convolutional module to the given input.
        '''
        if self.uds_type == 'downsample':
            if self.is_1d:
                # downsample only along first spatial dim
                h1 = dnn_pool(input, (2, 1), stride=(2, 1), mode='max', pad=(0, 0))
            else:
                # downsample along both spatial dims
                h1 = dnn_pool(input, (2, 2), stride=(2, 2), mode='max', pad=(0, 0))
        else:
            # upsample along first spatial dim
            h1 = T.repeat(input, 2, axis=2)
            if not self.is_1d:
                # upsample along second spatial dim for 2d inputs
                h1 = T.repeat(input, 2, axis=3)
        return h1


class BasicConvModuleNEW(object):
    '''
    Simple convolutional layer for use anywhere?

    Params:
        in_chans: number of channels in input
        out_chans: number of channels to produce as output
        filt_shape: filter shape, should be square and odd dim
        stride: whether to use 'double', 'single', or 'half' stride.
        is_1d: whether this module will operate on 1d sequences
        act_func: --
        mod_name: text name to identify this module in theano graph
    '''
    def __init__(self,
                 in_chans, out_chans,
                 filt_shape, stride='single',
                 is_1d=False,
                 act_func='ident',
                 mod_name='basic_conv'):
        assert (stride in ['single', 'double', 'half']), \
            "stride should be 'single', 'double', or 'half'."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.filt_dim = filt_shape[0]
        self.stride = stride
        self.is_1d = is_1d
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
        '''
        Initialize parameters for the layers in this module.
        '''
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        fd1 = self.filt_dim
        fd2 = 1 if self.is_1d else fd1
        self.w1 = weight_ifn((self.out_chans, self.in_chans, fd1, fd2),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.b1]
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input):
        '''
        Apply this convolutional module to the given input.
        '''
        if self.filt_dim == 2:
            bm = 0  # assume filt_dim == 2 means we're doing 2x downsampling
        else:
            bm = int((self.filt_dim - 1) / 2)
        if self.is_1d:
            # no padding or striding on second spatial dim when running in 1d
            if self.stride == 'single':
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, 0))
            elif self.stride == 'double':
                h1 = dnn_conv(input, self.w1, subsample=(2, 1), border_mode=(bm, 0))
            else:
                h1 = deconv(input, self.w1, subsample=(2, 1), border_mode=(bm, 0))
        else:
            # in 2d operation we pad and set stride for both spatial dimensions
            if self.stride == 'single':
                h1 = dnn_conv(input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
            elif self.stride == 'double':
                h1 = dnn_conv(input, self.w1, subsample=(2, 2), border_mode=(bm, bm))
            else:
                h1 = deconv(input, self.w1, subsample=(2, 2), border_mode=(bm, bm))
        # apply bias and activation function
        h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
        h1 = self.act_func(h1)
        return h1


class BasicConvGRUModuleRNN(object):
    '''
    Stateful convolutional module for use in various things.

    Parameters:
        state_chans: dimension of recurrent state in this module
        input_chans: dimension of input to this module
        spatial_shape: 2d spatial shape of this convolution
        filt_shape: 2d shape of the convolutional filters
        is_1d: whether this module operates on 1d input
        act_func: activation function to apply (should be tanh)
        mod_name: string name for this module
    '''
    def __init__(self,
                 state_chans, input_chans,
                 spatial_shape, filt_shape,
                 is_1d=False,
                 act_func='tanh', mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name'), \
            'module name is required.'
        self.state_chans = state_chans
        self.input_chans = input_chans
        self.spatial_shape = spatial_shape
        self.filt_dim = filt_shape[0]
        self.is_1d = is_1d
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
        self._init_params()  # initialize parameters
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        full_input_chans = self.state_chans + self.input_chans
        fd1 = self.filt_dim
        fd2 = 1 if self.is_1d else fd1
        # initialize gating parameters
        self.w1 = weight_ifn((2 * self.state_chans, full_input_chans, fd1, fd2),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize state update parameters
        self.w2 = weight_ifn((self.state_chans, full_input_chans, fd1, fd2),
                             "{}_w2".format(self.mod_name))
        self.b2 = bias_ifn((self.state_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        return

    def share_params(self, source_module):
        '''
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        '''
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.s0.set_value(floatX(param_dict['s0']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0, 'x', 'x')
        s0_zero = T.alloc(0., batch_size, self.state_chans,
                          self.spatial_shape[0], self.spatial_shape[1])
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply(self, state, input, rand_vals=None):
        '''
        Apply this GRU to an input and a previous state.
        '''
        bm1 = (self.filt_dim - 1) // 2  # use "same" mode convolutions
        bm2 = 0 if self.is_1d else bm1

        # compute update gate and remember gate
        gate_input = T.concatenate([state, input], axis=1)
        h = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm1, bm2))
        h = h + self.b1.dimshuffle('x', 0, 'x', 'x')
        h = hard_sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]
        r = h[:, self.state_chans:, :, :]

        # compute new state proposal
        update_input = T.concatenate([(r * state), input], axis=1)
        s = dnn_conv(update_input, self.w2, subsample=(1, 1), border_mode=(bm1, bm2))
        s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine old state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)
        return new_state


class GenConvGRUModuleRNN(object):
    '''
    Module for the top-down portion of a deep sequential conditional
    generative model.

    Parameters:
        state_chans: dimension of recurrent state in this module
        input_chans: dimension of TD inputs to this module
        rand_chans: dimension of latent variable inputs to this module
        spatial_shape: 2d spatial shape of this convolution
        filt_shape: 2d spatial shape of this layer's filters
        is_1d: whether this module will operate on 1D sequences
        act_func: activation function to apply (should be tanh)
        mod_name: string name for this module
    '''
    def __init__(self,
                 state_chans, input_chans, rand_chans,
                 spatial_shape, filt_shape,
                 is_1d=False,
                 act_func='relu', mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name'), \
            'module name is required.'
        self.state_chans = state_chans
        self.input_chans = input_chans
        self.rand_chans = rand_chans
        self.spatial_shape = spatial_shape
        self.filt_dim = filt_shape[0]
        self.is_1d = is_1d
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
        self._init_params()  # initialize parameters
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        full_input_chans = self.state_chans + self.input_chans + self.rand_chans
        # set spatial dim for the convolution filters
        fd1 = self.filt_dim
        fd2 = 1 if self.is_1d else self.filt_dim
        # w1/b1 are for gating function and w2/b2 are for the new state
        self.w1 = weight_ifn((2 * self.state_chans, full_input_chans, fd1, fd2),
                             "{}_w1".format(self.mod_name))
        self.w2 = weight_ifn((self.state_chans, full_input_chans, fd1, fd2),
                             "{}_w2".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.b2 = bias_ifn((self.state_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w1, self.w2, self.b1, self.b2])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        return

    def share_params(self, source_module):
        '''
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        '''
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.s0.set_value(floatX(param_dict['s0']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0, 'x', 'x')
        s0_zero = T.alloc(0., batch_size, self.state_chans,
                          self.spatial_shape[0], self.spatial_shape[1])
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply(self, state, input, rand_vals):
        '''
        Apply this GRU to an input and a previous state.
        '''
        # set border mode ("same" shape conv) for 2d/1d operation
        bm1 = (self.filt_dim - 1) // 2
        bm2 = 0 if self.is_1d else bm1

        # compute update gate and remember gate
        gate_input = T.concatenate([state, input, rand_vals], axis=1)
        h = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm1, bm2))
        h = h + self.b1.dimshuffle('x', 0, 'x', 'x')
        h = hard_sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]
        r = h[:, self.state_chans:, :, :]

        # compute new state proposal
        update_input = T.concatenate([(r * state), input, rand_vals], axis=1)
        s = dnn_conv(update_input, self.w2, subsample=(1, 1), border_mode=(bm1, bm2))
        s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine old state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)
        return new_state


class InfConvGRUModuleRNN(object):
    '''
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        state_chans: number of channels in the "info-merge" inputs to module
                     -- here, these provide the recurrent state
        td_state_chans: number of channels in recurrent TD state (from time t-1)
        td_input_chans: number of channels in the TD input (from time t)
        bu_chans: number of channels in the BU input (from time t)
        rand_chans: number of latent channels for which we we want conditionals
        spatial_shape: 2d spatial shape of this conv layer
        is_1d: whether this module will operate on 1d input
        act_func: ---
        use_td_cond: whether to condition on TD info
        mod_name: text name for identifying module in theano graph
    '''
    def __init__(self,
                 state_chans,
                 td_state_chans, td_input_chans,
                 bu_chans, rand_chans,
                 spatial_shape,
                 is_1d=False,
                 act_func='tanh', use_td_cond=False,
                 mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        self.td_state_chans = td_state_chans
        self.td_input_chans = td_input_chans
        self.bu_chans = bu_chans
        self.rand_chans = rand_chans
        self.spatial_shape = spatial_shape
        self.is_1d = is_1d
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
        self.use_td_cond = use_td_cond
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        td_in_chans = self.td_state_chans + self.td_input_chans
        full_in_chans = self.state_chans + td_in_chans + self.bu_chans
        # set spatial dimension for the convolutional filters
        fd1 = 3
        fd2 = 1 if self.is_1d else 3
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = weight_ifn((2 * self.state_chans, full_in_chans, fd1, fd2),
                                "{}_w1_im".format(self.mod_name))
        self.b1_im = bias_ifn((2 * self.state_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = weight_ifn((self.state_chans, full_in_chans, fd1, fd2),
                                "{}_w2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.state_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # initialize tranform from GRU to Gaussian conditional
        self.w3_im = weight_ifn((2 * self.rand_chans, self.state_chans, fd1, fd2),
                                "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((2 * self.rand_chans, td_in_chans, fd1, fd2),
                                    "{}_w1_td".format(self.mod_name))
            self.b1_td = bias_ifn((2 * self.rand_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.b1_td])
        return

    def share_params(self, source_module):
        '''
        Set this module to share parameters with source_module.
        '''
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = source_module.w1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = source_module.w2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.b2_im])
        # initialize conditioning parameters
        self.w3_im = source_module.w3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.b3_im])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.b1_td])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        self.s0.set_value(floatX(param_dict['s0']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0, 'x', 'x')
        s0_zero = T.alloc(0., batch_size, self.state_chans,
                          self.spatial_shape[0], self.spatial_shape[1])
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply_td(self, td_state, td_input):
        '''
        Put distributions over stuff based on td_state and td_input.
        '''
        bm1 = 1
        bm2 = 0 if self.is_1d else 1
        if self.use_td_cond:
            # simple conditioning on top-down input and recurrent state
            cond_input = T.concatenate([td_state, td_input], axis=1)
            h1 = dnn_conv(cond_input, self.w1_td, subsample=(1, 1), border_mode=(bm1, bm2))
            h1 = h1 + self.b1_td.dimshuffle('x', 0, 'x', 'x')
            out_mean = h1[:, :self.rand_chans, :, :]
            out_logvar = h1[:, self.rand_chans:, :, :]
        else:
            batch_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            rand_shape = (batch_size, self.rand_chans, rows, cols)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, state, td_state, td_input, bu_input):
        '''
        Combine prior IM state, prior TD state, td_input, and bu_input to update
        the IM state and to make a conditional Gaussian distribution.
        '''
        bm1 = 1
        bm2 = 0 if self.is_1d else 1
        # compute GRU gates
        gate_input = T.concatenate([state, td_state, td_input, bu_input], axis=1)
        h = dnn_conv(gate_input, self.w1_im, subsample=(1, 1), border_mode=(bm1, bm2))
        h = h + self.b1_im.dimshuffle('x', 0, 'x', 'x')
        h = hard_sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]  # state update gate
        r = h[:, self.state_chans:, :, :]  # state recall gate

        # compute new GRU state proposal
        update_input = T.concatenate([(r * state), td_state, td_input, bu_input], axis=1)
        s = dnn_conv(update_input, self.w2_im, subsample=(1, 1), border_mode=(bm1, bm2))
        s = s + self.b2_im.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine initial state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)

        # compute Gaussian conditional parameters
        g = dnn_conv(new_state, self.w3_im, subsample=(1, 1), border_mode=(bm1, bm2))
        g = g + self.b3_im.dimshuffle('x', 0, 'x', 'x')
        out_mean = g[:, :self.rand_chans, :, :]
        out_logvar = g[:, self.rand_chans:, :, :]
        return out_mean, out_logvar, new_state


class GenFCGRUModuleRNN(object):
    '''
    Module for the top-down portion of a deep sequential conditional
    generative model.

    Parameters:
        state_chans: dimension of recurrent state in this module
        input_chans: dimension of TD inputs to this module
        rand_chans: dimension of latent variable inputs to this module
        act_func: activation function to apply (should be tanh)
        mod_name: string name for this module
    '''
    def __init__(self,
                 state_chans, input_chans, rand_chans,
                 act_func='relu', mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name'), \
            'module name is required.'
        self.state_chans = state_chans
        self.input_chans = input_chans
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
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        full_input_chans = self.state_chans + self.input_chans + self.rand_chans
        # initialize gate layer parameters
        self.w1 = weight_ifn((full_input_chans, 2 * self.state_chans),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize gate layer parameters
        self.w2 = weight_ifn((full_input_chans, self.state_chans),
                             "{}_w2".format(self.mod_name))
        self.b2 = bias_ifn((self.state_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        return

    def share_params(self, source_module):
        '''
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        '''
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.s0.set_value(floatX(param_dict['s0']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0)
        s0_zero = T.alloc(0., batch_size, self.state_chans)
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply(self, state, input, rand_vals):
        '''
        Apply this GRU to an input and a previous state.
        '''
        # compute update gate and remember gate
        gate_input = T.concatenate([state, input, rand_vals], axis=1)
        h = T.dot(gate_input, self.w1)
        h = h + self.b1.dimshuffle('x', 0)
        h = hard_sigmoid(h + 1.)
        u = h[:, :self.state_chans]
        r = h[:, self.state_chans:]

        # compute new state proposal
        update_input = T.concatenate([(r * state), input, rand_vals], axis=1)
        s = T.dot(update_input, self.w2)
        s = s + self.b2.dimshuffle('x', 0)
        s = self.act_func(s)

        # combine old state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)
        return new_state


class InfFCGRUModuleRNN(object):
    '''
    Module for merging bottom-up and top-down information in a deep generative
    network with multiple layers of latent variables.

    Params:
        state_chans: number of channels in the "info-merge" inputs to module
                     -- here, these provide the recurrent state
        td_state_chans: number of channels in recurrent TD state (from time t-1)
        td_input_chans: number of channels in the TD input (from time t)
        bu_chans: number of channels in the BU input (from time t)
        rand_chans: number of latent channels for which we we want conditionals
        act_func: ---
        use_td_cond: whether to condition on TD info
        mod_name: text name for identifying module in theano graph
    '''
    def __init__(self,
                 state_chans,
                 td_state_chans, td_input_chans,
                 bu_chans, rand_chans,
                 act_func='tanh', use_td_cond=False,
                 mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        self.td_state_chans = td_state_chans
        self.td_input_chans = td_input_chans
        self.bu_chans = bu_chans
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
        self.use_td_cond = use_td_cond
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        td_in_chans = self.td_state_chans + self.td_input_chans
        full_in_chans = self.state_chans + td_in_chans + self.bu_chans
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = weight_ifn((full_in_chans, 2 * self.state_chans),
                                "{}_w1_im".format(self.mod_name))
        self.b1_im = bias_ifn((2 * self.state_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = weight_ifn((full_in_chans, self.state_chans),
                                "{}_w2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.state_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # initialize tranform from GRU to Gaussian conditional
        self.w3_im = weight_ifn((self.state_chans, 2 * self.rand_chans),
                                "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((td_in_chans, 2 * self.rand_chans),
                                    "{}_w1_td".format(self.mod_name))
            self.b1_td = bias_ifn((2 * self.rand_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.b1_td])
        return

    def share_params(self, source_module):
        '''
        Set this module to share parameters with source_module.
        '''
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = source_module.w1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = source_module.w2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.b2_im])
        # initialize conditioning parameters
        self.w3_im = source_module.w3_im
        self.b3_im = source_module.b3_im
        self.params.extend([self.w3_im, self.b3_im])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.b1_td])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        self.w3_im.set_value(floatX(param_dict['w3_im']))
        self.b3_im.set_value(floatX(param_dict['b3_im']))
        self.s0.set_value(floatX(param_dict['s0']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        param_dict['w3_im'] = self.w3_im.get_value(borrow=False)
        param_dict['b3_im'] = self.b3_im.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0)
        s0_zero = T.alloc(0., batch_size, self.state_chans)
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply_td(self, td_state, td_input):
        '''
        Put distributions over stuff based on td_state and td_input.
        '''
        if self.use_td_cond:
            # simple conditioning on top-down input and recurrent state
            td_state = T.flatten(td_state, 2)
            td_input = T.flatten(td_input, 2)
            cond_input = T.concatenate([td_state, td_input], axis=1)
            h1 = T.dot(cond_input, self.w1_td)
            h1 = h1 + self.b1_td.dimshuffle('x', 0)
            out_mean = h1[:, :self.rand_chans]
            out_logvar = h1[:, self.rand_chans:]
        else:
            batch_size = td_input.shape[0]
            rand_shape = (batch_size, self.rand_chans)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, state, td_state, td_input, bu_input):
        '''
        Combine prior IM state, prior TD state, td_input, and bu_input to update
        the IM state and to make a conditional Gaussian distribution.
        '''
        # flatten, in case we're sitting on top of some conv modulees
        state = T.flatten(state, 2)
        td_state = T.flatten(td_state, 2)
        td_input = T.flatten(td_input, 2)
        bu_input = T.flatten(bu_input, 2)

        # compute GRU gates
        gate_input = T.concatenate([state, td_state, td_input, bu_input], axis=1)
        h = T.dot(gate_input, self.w1_im)
        h = h + self.b1_im.dimshuffle('x', 0)
        h = hard_sigmoid(h + 1.)
        u = h[:, :self.state_chans]  # state update gate
        r = h[:, self.state_chans:]  # state recall gate

        # compute new GRU state proposal
        update_input = T.concatenate([(r * state), td_state, td_input, bu_input], axis=1)
        s = T.dot(update_input, self.w2_im)
        s = s + self.b2_im.dimshuffle('x', 0)
        s = self.act_func(s)

        # combine initial state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)

        # compute Gaussian conditional parameters
        g = T.dot(new_state, self.w3_im)
        g = g + self.b3_im.dimshuffle('x', 0)
        out_mean = g[:, :self.rand_chans]
        out_logvar = g[:, self.rand_chans:]
        return out_mean, out_logvar, new_state


class GenConvModuleNEW(object):
    '''
    Generator module that transforms a top-down input and some latent
    variables into an output.
    '''
    def __init__(self,
                 in_chans, out_chans, conv_chans, rand_chans,
                 filt_shape, act_func='relu',
                 mod_name='gm_conv'):
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv_chans = conv_chans
        self.rand_chans = rand_chans
        self.filt_dim = filt_shape[0]
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
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize first conv layer parameters
        self.w1 = weight_ifn((self.conv_chans, (self.in_chans + self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((self.conv_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize second conv layer parameters
        self.w2 = weight_ifn((self.out_chans, self.conv_chans, fd, fd),
                             "{}_w2".format(self.mod_name))
        self.b2 = bias_ifn((self.out_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        return

    def share_params(self, source_module):
        '''
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        '''
        self.params = []
        # share first conv layer parameters
        self.w1 = source_module.w1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.b1])
        # share second conv layer parameters
        self.w2 = source_module.w2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.b2.set_value(floatX(param_dict['b2']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        return param_dict

    def apply(self, input, rand_vals):
        '''
        Apply this generator module to some input.
        '''
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        full_input = T.concatenate([input, rand_vals], axis=1)
        # apply first conv layer
        h1 = dnn_conv(full_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        h1 = h1 + self.b1.dimshuffle('x', 0, 'x', 'x')
        h1 = self.act_func(h1)
        # apply second conv layer
        h2 = dnn_conv(h1, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        h2 = h2 + self.b2.dimshuffle('x', 0, 'x', 'x')
        h2 = self.act_func(h2)
        return h2


class InfConvMergeModuleNEW(object):
    '''
    Module for merging bottom-up and top-down information in a deep generative
    convolutional network with multiple layers of latent variables.

    Params:
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        rand_chans: number of latent channels that we want conditionals for
        conv_chans: number of channels in the "internal" convolution layer
        act_func: ---
        use_td_cond: whether to condition on TD info
        mod_name: text name for identifying module in theano graph
    '''
    def __init__(self,
                 td_chans, bu_chans, rand_chans, conv_chans,
                 act_func='relu', use_td_cond=False,
                 mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name')
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.rand_chans = rand_chans
        self.conv_chans = conv_chans
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
        self.use_td_cond = use_td_cond
        self.mod_name = mod_name
        self._init_params()
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize hidden conv layer params
        self.w1_im = weight_ifn((self.conv_chans, (self.td_chans + self.bu_chans), 3, 3),
                                "{}_w1_im".format(self.mod_name))
        self.b1_im = bias_ifn((self.conv_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.b1_im])
        # initialize conditional distribution params
        self.w2_im = weight_ifn((2 * self.rand_chans, self.conv_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.b2_im = bias_ifn((2 * self.rand_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((self.conv_chans, self.td_chans, 3, 3),
                                    "{}_w1_td".format(self.mod_name))
            self.b1_td = bias_ifn((self.conv_chans), "{}_b1_td".format(self.mod_name))
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = weight_ifn((2 * self.rand_chans, self.conv_chans, 3, 3),
                                    "{}_w2_td".format(self.mod_name))
            self.b2_td = bias_ifn((2 * self.rand_chans), "{}_b2_td".format(self.mod_name))
            self.params.extend([self.w2_td, self.b2_td])
        return

    def share_params(self, source_module):
        '''
        Set this module to share parameters with source_module.
        '''
        self.params = []
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize first conv layer parameters
        self.w1_im = source_module.w1_im
        self.b1_im = source_module.b1_im
        self.params.extend([self.w1_im, self.b1_im])
        # initialize second conv layer parameters
        self.w2_im = source_module.w2_im
        self.b2_im = source_module.b2_im
        self.params.extend([self.w2_im, self.b2_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = source_module.w1_td
            self.b1_td = source_module.b1_td
            self.params.extend([self.w1_td, self.b1_td])
            # initialize second conv layer parameters
            self.w2_td = source_module.w2_td
            self.b2_td = source_module.b2_td
            self.params.extend([self.w2_td, self.b2_td])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        # load info-merge parameters
        self.w1_im.set_value(floatX(param_dict['w1_im']))
        self.b1_im.set_value(floatX(param_dict['b1_im']))
        self.w2_im.set_value(floatX(param_dict['w2_im']))
        self.b2_im.set_value(floatX(param_dict['b2_im']))
        if self.use_td_cond:
            self.w1_td.set_value(floatX(param_dict['w1_td']))
            self.b1_td.set_value(floatX(param_dict['b1_td']))
            self.w2_td.set_value(floatX(param_dict['w2_td']))
            self.b2_td.set_value(floatX(param_dict['b2_td']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        # dump info-merge conditioning parameters
        param_dict['w1_im'] = self.w1_im.get_value(borrow=False)
        param_dict['b1_im'] = self.b1_im.get_value(borrow=False)
        param_dict['w2_im'] = self.w2_im.get_value(borrow=False)
        param_dict['b2_im'] = self.b2_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
            param_dict['w2_td'] = self.w2_td.get_value(borrow=False)
            param_dict['b2_td'] = self.b2_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input):
        '''
        Put distributions over stuff based on td_input.
        '''
        if self.use_td_cond:
            # compute a conditional distribution from the top-down input
            h1 = dnn_conv(td_input, self.w1_td, subsample=(1, 1), border_mode=(1, 1))
            h1 = h1 + self.b1_td.dimshuffle('x', 0, 'x', 'x')
            h1 = self.act_func(h1)
            h2 = dnn_conv(h1, self.w2_td, subsample=(1, 1), border_mode=(1, 1))
            h2 = h2 + self.b2_td.dimshuffle('x', 0, 'x', 'x')
            out_mean = h2[:, :self.rand_chans, :, :]
            out_logvar = h2[:, self.rand_chans:, :, :]
        else:
            batch_size = td_input.shape[0]
            rows = td_input.shape[2]
            cols = td_input.shape[3]
            rand_shape = (batch_size, self.rand_chans, rows, cols)
            out_mean = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                     dtype=theano.config.floatX)
            out_logvar = cu_rng.normal(size=rand_shape, avg=0.0, std=0.0001,
                                       dtype=theano.config.floatX)
        return out_mean, out_logvar

    def apply_im(self, td_input, bu_input):
        '''
        Combine td_input, bu_input, and im_input to compute stuff.
        '''
        # stack top-down and bottom-up inputs on top of each other
        full_input = T.concatenate([td_input, bu_input], axis=1)

        # apply hidden conv layer
        h1 = dnn_conv(full_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
        h1 = h1 + self.b1_im.dimshuffle('x', 0, 'x', 'x')
        h1 = self.act_func(h1)

        # apply conditonal distribution layer
        h2 = dnn_conv(h1, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
        h2 = h2 + self.b2_im.dimshuffle('x', 0, 'x', 'x')

        # get the conditional distribution parameters
        out_mean = h2[:, :self.rand_chans, :, :]
        out_logvar = h2[:, self.rand_chans:, :, :]
        return out_mean, out_logvar


class FCReshapeModule(object):
    '''
    Simple module for transforming and reshaping conv->fc or fc->conv.

    -- The transformation is always applied in fully-connected shape.

    Params:
        in_shape: shape of input, i.e., (rows, cols, feats) or (feats,)
        out_shape: shape of output, i.e., (rows, cols, feats) or (feats,)
        act_func: --
        mod_name: text name to identify this module in theano graph
    '''
    def __init__(self,
                 in_shape,
                 out_shape,
                 act_func='ident',
                 mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name')
        assert (len(in_shape) in [1, 2, 3])
        assert (len(out_shape) in [1, 2, 3])
        # get basic input/output shape information
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.out_reshape = False if (len(self.out_shape) == 1) else True
        self.in_chans = 1
        for c in self.in_shape:
            self.in_chans = self.in_chans * c
        self.out_chans = 1
        for c in self.out_shape:
            self.out_chans = self.out_chans * c
        # intialize remaining stuff
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
        '''
        Initialize parameters for the layers in this module.
        '''
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        self.w1 = weight_ifn((self.in_chans, self.out_chans),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((self.out_chans), "{}_b1".format(self.mod_name))
        self.params = [self.w1, self.b1]
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        return param_dict

    def apply(self, input):
        '''
        Flatten input, apply linear transform and activation, then reshape.
        '''
        h1 = T.dot(T.flatten(input, 2), self.w1)
        h1 = h1 + self.b1.dimshuffle('x', 0)
        h1 = self.act_func(h1)
        if self.out_reshape:
            if len(self.out_shape) == 2:
                output = h1.reshape((h1.shape[0], self.out_shape[0],
                                    self.out_shape[1]))
            elif len(self.out_shape) == 3:
                output = h1.reshape((h1.shape[0], self.out_shape[0],
                                    self.out_shape[1], self.out_shape[2]))
        else:
            output = h1
        return output


class TDModuleWrapperRNN(object):
    '''
    Wrapper around a generative TD module and an optional feedforward
    sequence of extra "post-processing" modules.

    Params:
        td_module: the first module in this TD module to apply. Inputs are
                    a previous state, a top-down input, and a latent input.
        mlp_modules: a list of the modules to apply to the output of td_module.
    '''
    def __init__(self, td_module, mlp_modules=None, use_shortcut=False,
                 use_rand=True, mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.td_module = td_module
        self.params = [p for p in td_module.params]
        if mlp_modules is not None:
            # use some extra post-processing modules
            self.mlp_modules = [m for m in mlp_modules]
            for mlp_mod in self.mlp_modules:
                self.params.extend(mlp_mod.params)
        else:
            # don't use any extra post-processing modules
            self.mlp_modules = None
        self.use_shortcut = use_shortcut
        self.use_rand = use_rand
        self.mod_name = mod_name
        return

    def dump_params(self):
        '''
        Dump params for later reloading by self.load_params.
        '''
        mod_param_dicts = [self.td_module.dump_params()]
        if self.mlp_modules is not None:
            mod_param_dicts.extend([m.dump_params() for m in self.mlp_modules])
        return mod_param_dicts

    def load_params(self, param_dict=None):
        '''
        Load params from the output of self.dump_params.
        '''
        self.td_module.load_params(param_dict=param_dict[0])
        if self.mlp_modules is not None:
            for pdict, mod in zip(param_dict[1:], self.mlp_modules):
                mod.load_params(param_dict=pdict)
        return

    def get_s0_for_batch(self, batch_size):
        '''
        Get an initial state for self.td_module.
        '''
        s0 = self.td_module.get_s0_for_batch(batch_size)
        return s0

    def apply(self, state, input, rand_vals):
        '''
        Process the recurrent td_module, then apply self.mlp_modules.
        -- this returns the updated recurrent state and an additional output
        '''
        state_new = self.td_module.apply(state, input, rand_vals)
        if self.mlp_modules is not None:
            # feedforward through the MLP modules to get a new output
            acts = None
            for mod in self.mlp_modules:
                if acts is None:
                    acts = [mod.apply(state_new)]
                else:
                    acts.append(mod.apply(acts[-1]))
            output = acts[-1]
        else:
            # use the updated recurrent state as output
            output = state_new
        return output, state_new


class TDModuleWrapperNEW(object):
    '''
    Wrapper around a generative TD module and an optional feedforward
    sequence of extra "post-processing" modules.
    -- This is for stateless modules.

    Params:
        td_module: the first module in this TD module to apply. Inputs are
                    a top-down input and a latent input.
        mlp_modules: a list of the modules to apply to the output of td_module.
    '''
    def __init__(self, td_module, mlp_modules=None, mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.td_module = td_module
        self.params = [p for p in td_module.params]
        if mlp_modules is not None:
            # use some extra post-processing modules
            self.mlp_modules = [m for m in mlp_modules]
            for mlp_mod in self.mlp_modules:
                self.params.extend(mlp_mod.params)
        else:
            # don't use any extra post-processing modules
            self.mlp_modules = None
        self.mod_name = mod_name
        return

    def dump_params(self):
        '''
        Dump params for later reloading by self.load_params.
        '''
        mod_param_dicts = [self.td_module.dump_params()]
        if self.mlp_modules is not None:
            mod_param_dicts.extend([m.dump_params() for m in self.mlp_modules])
        return mod_param_dicts

    def load_params(self, param_dict=None):
        '''
        Load params from the output of self.dump_params.
        '''
        self.td_module.load_params(param_dict=param_dict[0])
        if self.mlp_modules is not None:
            for pdict, mod in zip(param_dict[1:], self.mlp_modules):
                mod.load_params(param_dict=pdict)
        return

    def apply(self, input, rand_vals):
        '''
        Process the td_module, then apply self.mlp_modules.
        '''
        h1 = self.td_module.apply(input, rand_vals)
        if self.mlp_modules is not None:
            # feedforward through the MLP modules to get a new output
            acts = None
            for mod in self.mlp_modules:
                if acts is None:
                    acts = [mod.apply(h1)]
                else:
                    acts.append(mod.apply(acts[-1]))
            output = acts[-1]
        else:
            # use the updated recurrent state as output
            output = h1
        return output


class BUModuleWrapperRNN(object):
    '''
    Wrapper around a recurrent BU module and an optional feedforward
    sequence of extra "post-processing" modules.

    Params:
        bu_module: the first module in this BU module to apply. Inputs are
                    a previous state, a top-down input, and a latent input.
        mlp_modules: a list of the modules to apply to the output of bu_module.
    '''
    def __init__(self, bu_module, mlp_modules=None, use_shortcut=False,
                 mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.bu_module = bu_module
        self.params = [p for p in bu_module.params]
        if mlp_modules is not None:
            # use some extra post-processing modules
            self.mlp_modules = [m for m in mlp_modules]
            for mlp_mod in self.mlp_modules:
                self.params.extend(mlp_mod.params)
        else:
            # don't use any extra post-processing modules
            self.mlp_modules = None
        self.use_shortcut = use_shortcut
        self.mod_name = mod_name
        return

    def dump_params(self):
        '''
        Dump params for later reloading by self.load_params.
        '''
        mod_param_dicts = [self.bu_module.dump_params()]
        if self.mlp_modules is not None:
            mod_param_dicts.extend([m.dump_params() for m in self.mlp_modules])
        return mod_param_dicts

    def load_params(self, param_dict=None):
        '''
        Load params from the output of self.dump_params.
        '''
        self.bu_module.load_params(param_dict=param_dict[0])
        if self.mlp_modules is not None:
            for pdict, mod in zip(param_dict[1:], self.mlp_modules):
                mod.load_params(param_dict=pdict)
        return

    def get_s0_for_batch(self, batch_size):
        '''
        Get an initial state for self.bu_module.
        '''
        s0 = self.bu_module.get_s0_for_batch(batch_size)
        return s0

    def apply(self, state, input):
        '''
        Process the recurrent bu_module, then apply self.mlp_modules.
        -- this returns the updated recurrent state and an additional output
        '''
        if self.mlp_modules is not None:
            # feedforward through the MLP modules to get a new output
            acts = None
            for mod in self.mlp_modules:
                if acts is None:
                    acts = [mod.apply(input)]
                else:
                    acts.append(mod.apply(acts[-1]))
        else:
            # use the updated recurrent state as output
            acts = [input]
        output = self.bu_module.apply(state, acts[-1])
        return output, output


class IMModuleWrapperNEW(object):
    '''
    Wrapper around a conditional IM module and an optional feedforward
    sequence of extra "pre-processing" modules.
    -- This is for stateless modules.

    Params:
        inf_module: the final module in this IM module to apply. This takes a
                    bottom-up input, from the mlp_modules or direct from user.
        mlp_modules: a list of modules to apply to get an input for inf_module.
    '''
    def __init__(self, inf_module, mlp_modules=None, mod_name='no_name'):
        assert not (mod_name == 'no_name')
        self.inf_module = inf_module
        self.rand_chans = self.inf_module.rand_chans
        self.params = [p for p in inf_module.params]
        if mlp_modules is not None:
            # use some extra post-processing modules
            self.mlp_modules = [m for m in mlp_modules]
            for mlp_mod in self.mlp_modules:
                self.params.extend(mlp_mod.params)
        else:
            # don't use any extra post-processing modules
            self.mlp_modules = None
        self.mod_name = mod_name
        return

    def dump_params(self):
        '''
        Dump params for later reloading by self.load_params.
        '''
        mod_param_dicts = [self.inf_module.dump_params()]
        if self.mlp_modules is not None:
            mod_param_dicts.extend([m.dump_params() for m in self.mlp_modules])
        return mod_param_dicts

    def load_params(self, param_dict=None):
        '''
        Load params from the output of self.dump_params.
        '''
        self.inf_module.load_params(param_dict=param_dict[0])
        if self.mlp_modules is not None:
            for pdict, mod in zip(param_dict[1:], self.mlp_modules):
                mod.load_params(param_dict=pdict)
        return

    def apply_td(self, td_input):
        '''
        Apply the underlying inf module's `apply_td`.
        '''
        cond_mean, cond_logvar = self.inf_module.apply_td(td_input)
        return cond_mean, cond_logvar

    def apply_im(self, bu_input, td_input=None):
        '''
        Process self.mlp_modules, then apply self.inf_module.
        '''
        if self.mlp_modules is not None:
            # feedforward through the MLP modules to get a new output
            acts = None
            for mod in self.mlp_modules:
                if acts is None:
                    acts = [mod.apply(bu_input)]
                else:
                    acts.append(mod.apply(acts[-1]))
            mlp_out = acts[-1]
        else:
            # use the updated recurrent state as output
            mlp_out = bu_input
        if td_input is None:
            td_input = mlp_out
        cond_mean, cond_logvar = self.inf_module.apply_im(mlp_out, td_input)
        return cond_mean, cond_logvar


class ContextualGRU(object):
    '''
    Recurrent decoder with input from the "primary" sequence and from a
    "context" sequence.

    Parameters:
        state_chans: dimension of recurrent state
        input_chans: dimension of input sequence
        output_chans: dimension for the output from this module
        context_chans: dimension of additional "context" input
        act_func: activation function to apply (should be tanh)
        mod_name: string name for this module
    '''
    def __init__(self,
                 state_chans, input_chans, output_chans, context_chans,
                 act_func='relu', mod_name='no_name'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        assert not (mod_name == 'no_name'), \
            'module name is required.'
        self.state_chans = state_chans
        self.input_chans = input_chans
        self.output_chans = output_chans
        self.context_chans = context_chans
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
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Orthogonal()
        bias_ifn = inits.Constant(c=0.)
        full_input_chans = self.state_chans + self.input_chans + self.context_chans
        # initialize gating parameters
        self.w1 = weight_ifn((full_input_chans, 2 * self.state_chans),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize state update parameters
        self.w2 = weight_ifn((full_input_chans, self.state_chans),
                             "{}_w2".format(self.mod_name))
        self.b2 = bias_ifn((self.state_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.b2])
        # initialize output funcction parameters
        self.w3 = weight_ifn((self.state_chans, self.output_chans),
                             "{}_w3".format(self.mod_name))
        self.b3 = bias_ifn((self.output_chans), "{}_b3".format(self.mod_name))
        self.params.extend([self.w3, self.b3])
        # initialize trainable initial state
        self.s0 = bias_ifn((self.state_chans), "{}_s0".format(self.mod_name))
        self.params.extend([self.s0])
        return

    def share_params(self, source_module):
        '''
        Set parameters in this module to be shared with source_module.
        -- This just sets our parameter info to point to the shared variables
           used by source_module.
        '''
        self.params = []
        # share gating parameters
        self.w1 = source_module.w1
        self.b1 = source_module.b1
        self.params.extend([self.w1, self.b1])
        # share state update parameters
        self.w2 = source_module.w2
        self.b2 = source_module.b2
        self.params.extend([self.w2, self.b2])
        # share output function parameters
        self.w3 = source_module.w3
        self.b3 = source_module.b3
        self.params.extend([self.w3, self.b3])
        # share initial state
        self.s0 = source_module.s0
        self.params.extend([self.s0])
        return

    def load_params(self, param_dict):
        '''
        Load module params directly from a dict of numpy arrays.
        '''
        self.w1.set_value(floatX(param_dict['w1']))
        self.b1.set_value(floatX(param_dict['b1']))
        self.w2.set_value(floatX(param_dict['w2']))
        self.b2.set_value(floatX(param_dict['b2']))
        self.w3.set_value(floatX(param_dict['w3']))
        self.b3.set_value(floatX(param_dict['b3']))
        self.s0.set_value(floatX(param_dict['s0']))
        return

    def dump_params(self):
        '''
        Dump module params directly to a dict of numpy arrays.
        '''
        param_dict = {}
        param_dict['w1'] = self.w1.get_value(borrow=False)
        param_dict['b1'] = self.b1.get_value(borrow=False)
        param_dict['w2'] = self.w2.get_value(borrow=False)
        param_dict['b2'] = self.b2.get_value(borrow=False)
        param_dict['w3'] = self.w3.get_value(borrow=False)
        param_dict['b3'] = self.b3.get_value(borrow=False)
        param_dict['s0'] = self.s0.get_value(borrow=False)
        return param_dict

    def get_s0_for_batch(self, batch_size):
        '''
        Get initial state for this module, for minibatch of given size.
        '''
        # broadcast self.s0 into the right shape for this batch
        s0_init = self.s0.dimshuffle('x', 0)
        s0_zero = T.alloc(0., batch_size, self.state_chans)
        s0_batch = s0_init + s0_zero
        return s0_batch

    def apply(self, input, context):
        '''
        Apply this GRU to an input sequence and a context sequence.

        -- input and context should be shapes:
             (nbatch, seq_len, self.input_chans)
             (nbatch, seq_len, self.context_chans)
        '''

        def _step_func(x_in, x_ct, s_i):
            # compute output for the current state
            o_i = T.dot(s_i, self.w3) + self.b3.dimshuffle('x', 0)
            # o_i = clip_softmax(o_i, axis=1)

            # compute update gate and remember gate
            gate_input = T.concatenate([x_in, x_ct, s_i], axis=1)
            h = T.dot(gate_input, self.w1) + self.b1.dimshuffle('x', 0)
            h = hard_sigmoid(h + 1.)
            u = h[:, :self.state_chans]
            r = h[:, self.state_chans:]

            # compute new state proposal
            state_input = T.concatenate([x_in, x_ct, (r * s_i)], axis=1)
            s_new = T.dot(state_input, self.w2) + self.b2.dimshuffle('x', 0)
            s_new = self.act_func(s_new)

            # combine old state and proposed new state based on u
            s_ip1 = (u * s_i) + ((1. - u) * s_new)
            return s_ip1, o_i

        # shuffle inputs to have sequence dimension first
        seq_input = input.dimshuffle(1, 0, 2)
        seq_context = context.dimshuffle(1, 0, 2)
        # get initial state for GRU
        s0_full = self.get_s0_for_batch(batch_size=input.shape[0])

        # apply scan over the input sequences, to generate an output sequence
        init_vals = [s0_full, None]
        scan_results, scan_updates = \
            theano.scan(_step_func,
                        outputs_info=init_vals,
                        sequences=[seq_input, seq_context])
        # shuffle output back to shape: (nbatch, seq_len, self.output_chans)
        output = scan_results[1].dimshuffle(1, 0, 2)
        return output, scan_updates


##############
# EYE BUFFER #
##############
