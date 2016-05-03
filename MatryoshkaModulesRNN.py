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
    Convolutional GRU, takes an "exogenous" input and a "recurrent" input.
    '''
    def __init__(self, filt_shape, in_chans, state_chans,
                 act_func='tanh', stride='single',
                 mod_name='gm_conv'):
        assert (filt_shape == (3, 3) or filt_shape == (5, 5)), \
            "filt_shape must be (3, 3) or (5, 5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
            "invalid act_func {}.".format(act_func)
        self.filt_dim = filt_shape[0]
        self.in_chans = in_chans
        self.state_chans = state_chans
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
        update_input = T.concatenate([(r * state), input])
        s = dnn_conv(update_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        s = s + self.b2.dimshuffle('x', 0, 'x', 'x')
        s = self.act_func(s)

        # combine old state and proposed new state based on u
        new_state = (u * state) + ((1. - u) * s)
        return new_state


class GenConvGRUModuleRNN(object):
    """
    Test module.
    """
    def __init__(self,
                 in_chans, out_chans, rand_chans, filt_shape,
                 use_rand=True,
                 unif_drop=0.0, chan_drop=0.0, apply_bn=True,
                 use_bn_params=True, act_func='relu',
                 mod_name='gm_conv'):
        assert ((in_chans == out_chans)), \
                "in_chans == out_chans is required."
        assert (filt_shape == (3,3) or filt_shape == (5,5)), \
                "filt_shape must be (3,3) or (5,5)."
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.rand_chans = rand_chans
        self.filt_dim = filt_shape[0]
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
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
        gain_ifn = inits.Normal(loc=1., scale=0.02)
        bias_ifn = inits.Constant(c=0.)
        fd = self.filt_dim
        # initialize gate layer parameters
        self.w1 = weight_ifn((self.in_chans, (self.in_chans+self.rand_chans), fd, fd),
                             "{}_w1".format(self.mod_name))
        self.g1 = gain_ifn((self.in_chans), "{}_g1".format(self.mod_name))
        self.b1 = bias_ifn((self.in_chans), "{}_b1".format(self.mod_name))
        self.params.extend([self.w1, self.g1, self.b1])

        # initialize gate layer parameters
        self.w2 = weight_ifn((self.in_chans, (self.in_chans+self.rand_chans), fd, fd),
                             "{}_w2".format(self.mod_name))
        self.g2 = gain_ifn((self.in_chans), "{}_g2".format(self.mod_name))
        self.b2 = bias_ifn((self.in_chans), "{}_b2".format(self.mod_name))
        self.params.extend([self.w2, self.g2, self.b2])

        # initialize first new state layer parameters
        self.w3 = weight_ifn((self.in_chans, (self.in_chans+self.rand_chans), fd, fd),
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
        rand_shape = rand_vals.shape # return vals must be theano vars

        # compute update gate and remember gate
        gate_input = T.concatenate([rand_vals, input], axis=1)
        h1 = dnn_conv(gate_input, self.w1, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h1 = switchy_bn(h1, g=self.g1, b=self.b1, n=noise,
                           use_gb=self.use_bn_params)
        else:
            h1 = h1 + self.b1.dimshuffle('x',0,'x','x')
            h1 = add_noise(h1, noise=noise)
        u = sigmoid(h1 + 1.)
        #
        h2 = dnn_conv(gate_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            h2 = switchy_bn(h2, g=self.g2, b=self.b2, n=noise,
                           use_gb=self.use_bn_params)
        else:
            h2 = h2 + self.b2.dimshuffle('x',0,'x','x')
            h2 = add_noise(h2, noise=noise)
        r = sigmoid(h2 + 1.)
        # compute new state proposal -- include hidden layer
        state_input = T.concatenate([rand_vals, r*input], axis=1)
        s = dnn_conv(state_input, self.w2, subsample=(1, 1), border_mode=(bm, bm))
        if self.apply_bn:
            s = switchy_bn(s, g=self.g2, b=self.b2, n=noise,
                           use_gb=self.use_bn_params)
        else:
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


class InfConvGRUModuleRNN(object):
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
                 act_func='tanh',
                 unif_drop=0.0,
                 chan_drop=0.0,
                 apply_bn=False,
                 use_td_cond=False,
                 use_bn_params=True,
                 mod_name='gm_conv'):
        assert (act_func in ['ident', 'tanh', 'relu', 'lrelu', 'elu']), \
                "invalid act_func {}.".format(act_func)
        self.td_chans = td_chans
        self.bu_chans = bu_chans
        self.im_chans = im_chans
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
        self.chan_drop = chan_drop
        self.apply_bn = apply_bn
        self.use_td_cond = use_td_cond
        self.use_bn_params = True
        self.mod_name = mod_name
        self._init_params() # initialize parameters
        return

    def _init_params(self):
        """
        Initialize parameters for the layers in this module.
        """
        self.params = []
        weight_ifn = inits.Normal(loc=0., scale=0.02)
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