import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import inits
from lib.rng import cu_rng
from lib.theano_utils import floatX
from MatryoshkaModules import *

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify(leak=0.1)
bce = T.nnet.binary_crossentropy
hard_sigmoid = T.nnet.hard_sigmoid
tanh = activations.Tanh()
elu = activations.ELU()


class BasicConvGRUModuleRNN(object):
    '''
    Convolutional GRU, takes a recurrent input and an exogenous input.
    '''
    def __init__(self, state_chans, in_chans, filt_shape,
                 act_func='tanh', mod_name='gm_conv'):
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        self.in_chans = in_chans
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
        self._init_params()  # initialize parameters
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        full_in_chans = self.state_chans + self.in_chans
        # initialize gating parameters
        self.w1 = weight_ifn((2 * self.state_chans, full_in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize state update parameters
        self.w2 = weight_ifn((self.state_chans, full_in_chans, fd, fd),
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

    def apply(self, state, input, rand_vals=None):
        '''
        Apply this GRU to an input and a previous state.
        '''
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        # compute update gate and remember gate
        gate_input = T.concatenate([state, input], axis=1)
        h = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        h = h + self.b1.dimshuffle('x', 0, 'x', 'x')
        h = sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]
        r = h[:, self.state_chans:, :, :]

        # compute new state proposal
        update_input = T.concatenate([(r * state), input], axis=1)
        s = dnn_conv(update_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine old state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)
        return new_state


class GenConvGRUModuleRNN(object):
    '''
    This model receives a recurrent input, a top-down input, and a set of
    stochastic latent variables.
    '''
    def __init__(self,
                 state_chans, in_chans, rand_chans, filt_shape,
                 act_func='relu', mod_name='gm_conv'):
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        self.in_chans = in_chans
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
        self._init_params()  # initialize parameters
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        full_in_chans = self.state_chans + self.in_chans + self.rand_chans
        # initialize gate layer parameters
        self.w1 = weight_ifn((2 * self.state_chans, full_in_chans, fd, fd),
                             "{}_w1".format(self.mod_name))
        self.b1 = bias_ifn((2 * self.state_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.b1])
        # initialize gate layer parameters
        self.w2 = weight_ifn((self.state_chans, full_in_chans, fd, fd),
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

    def apply(self, state, input, rand_vals):
        '''
        Apply this GRU to an input and a previous state.
        '''
        bm = (self.filt_dim - 1) // 2  # use "same" mode convolutions

        # compute update gate and remember gate
        gate_input = T.concatenate([state, input, rand_vals], axis=1)
        h = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        h = h + self.b1.dimshuffle('x', 0, 'x', 'x')
        h = sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]
        r = h[:, self.state_chans:, :, :]

        # compute new state proposal
        update_input = T.concatenate([(r * state), input, rand_vals], axis=1)
        s = dnn_conv(update_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
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
        td_chans: number of channels in the "top-down" inputs to module
        bu_chans: number of channels in the "bottom-up" inputs to module
        rand_chans: number of latent channels for which we we want conditionals
        act_func: ---
        use_td_cond: whether to condition on TD info
        mod_name: text name for identifying module in theano graph
    '''
    def __init__(self,
                 state_chans, td_chans, bu_chans, rand_chans,
                 act_func='tanh', use_td_cond=False,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.state_chans = state_chans
        self.td_chans = td_chans
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
        self._init_params()  # initialize parameters
        return

    def _init_params(self):
        '''
        Initialize parameters for the layers in this module.
        '''
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        full_in_chans = self.state_chans + self.td_chans + self.bu_chans
        ############################################
        # Initialize "inference" model parameters. #
        ############################################
        # initialize GRU gating parameters
        self.w1_im = weight_ifn((2 * self.state_chans, full_in_chans, 3, 3),
                                "{}_w1_im".format(self.mod_name))
        self.b1_im = bias_ifn((2 * self.state_chans), "{}_b1_im".format(self.mod_name))
        self.params.extend([self.w1_im, self.b1_im])
        # initialize GRU state update parameters
        self.w2_im = weight_ifn((self.state_chans, full_in_chans, 3, 3),
                                "{}_w2_im".format(self.mod_name))
        self.b2_im = bias_ifn((self.state_chans), "{}_b2_im".format(self.mod_name))
        self.params.extend([self.w2_im, self.b2_im])
        # initialize tranform from GRU to Gaussian conditional
        self.w3_im = weight_ifn((2 * self.rand_chans, self.state_chans, 3, 3),
                                "{}_w3_im".format(self.mod_name))
        self.b3_im = bias_ifn((2 * self.rand_chans), "{}_b3_im".format(self.mod_name))
        self.params.extend([self.w3_im, self.b3_im])
        # initialize trainable initial state
        self.s0_im = bias_ifn((self.state_chans), "{}_s0_im".format(self.mod_name))
        self.params.extend([self.s0_im])
        # setup params for implementing top-down conditioning
        if self.use_td_cond:
            self.w1_td = weight_ifn((2 * self.rand_chans, self.td_chans, 3, 3),
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
        self.s0_im = source_module.s0_im
        self.params.extend([self.s0_im])
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
        self.s0_im.set_value(floatX(param_dict['s0_im']))
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
        param_dict['s0_im'] = self.s0_im.get_value(borrow=False)
        if self.use_td_cond:
            param_dict['w1_td'] = self.w1_td.get_value(borrow=False)
            param_dict['b1_td'] = self.b1_td.get_value(borrow=False)
        return param_dict

    def apply_td(self, td_input):
        '''
        Put distributions over stuff based on td_input.
        '''
        if self.use_td_cond:
            # simple linear conditioning on top-down state
            h1 = dnn_conv(td_input, self.w1_td, subsample=(1, 1), border_mode=(1, 1))
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

    def apply_im(self, state, td_input, bu_input):
        '''
        Combine prior IM state, td_input, and bu_input to update the IM state
        and to make a conditional Gaussian distribution.
        '''
        # compute GRU gates
        gate_input = T.concatenate([state, td_input, bu_input], axis=1)
        h = dnn_conv(gate_input, self.w1_im, subsample=(1, 1), border_mode=(1, 1))
        h = h + self.b1_im.dimshuffle('x', 0, 'x', 'x')
        h = sigmoid(h + 1.)
        u = h[:, :self.state_chans, :, :]  # state update gate
        r = h[:, self.state_chans:, :, :]  # state recall gate

        # compute new GRU state proposal
        update_input = T.concatenate([(r * state), td_input, bu_input], axis=1)
        s = dnn_conv(update_input, self.w2_im, subsample=(1, 1), border_mode=(1, 1))
        s = s + self.b2_im.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine initial state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)

        # compute Gaussian conditional parameters
        g = dnn_conv(new_state, self.w3_im, subsample=(1, 1), border_mode=(1, 1))
        g = g + self.b3_im.dimshuffle('x', 0, 'x', 'x')
        out_mean = g[:, :self.rand_chans, :, :]
        out_logvar = g[:, self.rand_chans:, :, :]
        return out_mean, out_logvar, new_state





##############
# EYE BUFFER #
##############
