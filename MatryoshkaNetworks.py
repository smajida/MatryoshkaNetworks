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



class GeneratorNetwork(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    sequence of modules from MatryoshkaModules.py
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
        self.generate_samples = self._construct_generate_samples()
        return

    def apply(self, rand_vals=None, batch_size=None, return_acts=False):
        """
        Apply this generator network using the given random values.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
                "need either batch_size or rand_vals"
        assert ((batch_size is None) or (rand_vals is None)), \
                "need either batch_size or rand_vals"
        assert ((len(rand_vals) == len(self.modules)) or (rand_vals is None)), \
                "random values should be appropriate for this network."
        if rand_vals is None:
            rand_vals = [None for i in range(len(self.modules))]
        acts = []
        for i, rvs in enumerate(rand_vals):
            if i == 0:
                # first module takes no inputs
                acts.append(self.modules[i].apply(rand_vals=rvs,
                                                  batch_size=batch_size))
            else:
                # subsequent modules take earlier outputs as inputs
                acts.append(self.modules[i].apply(acts[-1], rand_vals=rvs))
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
