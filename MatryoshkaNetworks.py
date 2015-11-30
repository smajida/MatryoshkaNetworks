import numpy as np
import numpy.random as npr
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
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, deconv
from lib.theano_utils import floatX, sharedX

#
# Phil's business
#
from MatryoshkaModules import DiscConvModule, DiscFCModule, GenConvModule, \
                              GenFCModule, BasicConvModule



class GenNetwork(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    sequence of modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this GenNetwork. The
                 first module must be an instance of GenFCModule and the
                 remaining modules should be instances of GenConvModule or
                 BasicConvModule.
        output_transform: transform to apply to output of final convolutional
                          module to get the output of this GenNetwork.
    """
    def __init__(self, modules, output_transform):
        self.modules = [m for m in modules]
        self.fc_module = self.modules[0]
        self.conv_modules = self.modules[1:]
        self.rng = RandStream(123)
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        self.output_transform = output_transform
        # construct a theano function for drawing samples from this model
        print("Compiling sample generator...")
        self.generate_samples = self._construct_generate_samples()
        print("DONE.")
        return

    def apply(self, rand_vals=None, batch_size=None, return_acts=False):
        """
        Apply this generator network using the given random values.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
                "need _either_ batch_size or rand_vals."
        assert ((batch_size is None) or (rand_vals is None)), \
                "need _either_ batch_size or rand_vals."
        assert ((rand_vals is None) or (len(rand_vals) == len(self.modules))), \
                "random values should be appropriate for this network."
        if rand_vals is None:
            # no random values were provided, which means we'll be generating
            # based on a user-provided batch_size.
            rand_vals = [None for i in range(len(self.modules))]
        else:
            if rand_vals[0] is None:
                # random values were provided, but not for the fc module, so we
                # need the batch size so that the fc module produces output
                # with the appropriate shape.
                rand_vals[0] = -1
        acts = []
        for i, rvs in enumerate(rand_vals):
            if i == 0:
                # feedforward through the fc module
                if not (rvs == -1):
                    # rand_vals was not given or rand_vals[0] was given...
                    acts.append(self.modules[i].apply(rand_vals=rvs,
                                                      batch_size=batch_size))
                else:
                    # rand_vals was given, but rand_vals[0] was not given...
                    # we need to get the batch_size param for this feedforward
                    _rand_vals = [v for v in rand_vals if not (v is None)]
                    bs = _rand_vals[0].shape[0]
                    acts.append(self.modules[i].apply(rand_vals=None,
                                                      batch_size=bs))
            else:
                # feedforward through a convolutional module
                acts.append(self.modules[i].apply(acts[-1], rand_vals=rvs))
        # apply final transform (e.g. tanh or sigmoid) to final activations
        output = self.output_transform(acts[-1])
        if return_acts:
            result = [output, acts]
        else:
            result = output
        return result

    def _construct_generate_samples(self):
        """
        Generate some samples from this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_samples = self.apply(batch_size=batch_size)
        # compile a theano function for computing the stochastic feedforward
        sample_func = theano.function([batch_size], sym_samples)
        def sampler(sample_count):
            samples = sample_func(sample_count)
            return samples
        return sampler
