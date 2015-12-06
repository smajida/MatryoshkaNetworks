import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from collections import OrderedDict

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, deconv
from lib.theano_utils import floatX, sharedX

#
# Phil's business
#
from LogPDFs import log_prob_gaussian2, gaussian_kld
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
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        self.output_transform = output_transform
        # construct a theano function for drawing samples from this model
        print("Compiling sample generator...")
        self.generate_samples = self._construct_generate_samples()
        samps = self.generate_samples(50)
        print("DONE.")
        print("Compiling rand shape computer...")
        self.compute_rand_shapes = self._construct_compute_rand_shapes()
        shapes = self.compute_rand_shapes(50)
        print("DONE.")
        return

    def apply(self, rand_vals=None, batch_size=None, return_acts=False,
              rand_shapes=False):
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
        r_shapes = []
        res = None
        for i, rvs in enumerate(rand_vals):
            if i == 0:
                # feedforward through the fc module
                if not (rvs == -1):
                    # rand_vals was not given or rand_vals[0] was given...
                    res = self.modules[i].apply(rand_vals=rvs,
                                                batch_size=batch_size,
                                                rand_shapes=rand_shapes)
                else:
                    # rand_vals was given, but rand_vals[0] was not given...
                    # we need to get the batch_size param for this feedforward
                    _rand_vals = [v for v in rand_vals if not (v is None)]
                    bs = _rand_vals[0].shape[0]
                    res = self.modules[i].apply(rand_vals=None,
                                                batch_size=bs,
                                                rand_shapes=rand_shapes)
            else:
                # feedforward through a convolutional module
                res = self.modules[i].apply(acts[-1], rand_vals=rvs,
                                            rand_shapes=rand_shapes)
            if not rand_shapes:
                acts.append(res)
            else:
                acts.append(res[0])
                r_shapes.append(res[1])
        # apply final transform (e.g. tanh or sigmoid) to final activations
        output = self.output_transform(acts[-1])
        if return_acts:
            result = [output, acts]
        else:
            result = output
        if rand_shapes:
            result = r_shapes
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
        return sample_func

    def _construct_compute_rand_shapes(self):
        """
        Compute the shape of stochastic input for all layers in this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_shapes = self.apply(batch_size=batch_size, rand_shapes=True)
        # compile a theano function for computing the stochastic feedforward
        shape_func = theano.function([batch_size], sym_shapes)
        return shape_func


class VarInfModel(object):
    """
    Wrapper class for extracting variational estimates of log-likelihood from
    a GenNetwork. The VarInfModel will be specific to a given set of inputs.
    """
    def __init__(self, X, M, gen_network, mean_init_func, logvar_init_func):
        # observations for which to perform variational inference
        self.X = sharedX(X, name='VIM.X')
        # masks for which components of each observation are "visible"
        self.M = sharedX(M, name='VIM.M')
        self.gen_network = gen_network
        self.obs_count = X.shape[0]
        self.mean_init_func = mean_init_func
        self.logvar_init_func = logvar_init_func
        # get initial means and log variances of stochastic variables for the
        # observations in self.X, using the GenNetwork self.gen_network. also,
        # make symbolic random variables for passing to self.gen_network.
        self.rv_means, self.rv_logvars, self.rand_vals = self._construct_rvs()
        # get samples from self.gen_network using self.rv_mean/self.rv_logvar
        self.Xg = self.gen_network.apply(rand_vals=self.rand_vals)
        # self.output_logvar modifies the output distribution
        self.output_logvar = sharedX(np.zeros((1,)), name='VIM.output_logvar')
        self.bounded_logvar = 8.0 * T.tanh(self.output_logvar[0] / 8.0)
        # compute reconstruction/NLL cost using self.Xg
        self.nlls = self._construct_nlls(x=self.X, m=self.M, x_hat=self.Xg,
                                         out_logvar=self.bounded_logvar)
        # construct symbolic vars for KL divergences between our reparametrized
        # Gaussians, and some ZMUV Gaussians.
        self.lam_kld = sharedX(np.ones((1,)), name='VIM.lam_kld')
        self.set_lam_kld(lam_kld=1.0)
        self.klds = self._construct_klds()
        # make symbolic vars for the overall optimization objective
        self.vfe_bounds = self.nlls + self.klds
        self.opt_cost = T.mean(self.nlls) + (self.lam_kld[0]*T.mean(self.klds))
        # construct updates for self.rv_means/self.rv_logvars
        self.params = self.rv_means + self.rv_logvars + [self.output_logvar]
        self.lr = T.scalar()
        updater = updates.Adam(lr=self.lr, b1=0.5, b2=0.98, e=1e-4)
        print("Constructing VarInfModel updates...")
        self.param_updates = updater(self.params, self.opt_cost)
        print("Compiling VarInfModel.train()...")
        self.train = theano.function(inputs=[self.lr],
                                     outputs=[self.opt_cost, self.vfe_bounds],
                                     updates=self.param_updates)
        # construct theano function for estimating log-likelihood cost given
        # the observations in self.X and the current means/logvars in
        # self.rv_means/self.rv_logvars.
        print("Compiling VarInfModel.sample_vfe_bounds()...")
        self.sample_vfe_bounds = theano.function(inputs=[],
                                                 outputs=self.vfe_bounds)
        # construct theano function for sampling "reconstructions" from
        # self.gen_network, given self.rv_means/self.rv_logvars
        print("Compiling VarInfModel.sample_Xg()...")
        self.sample_Xg = theano.function(inputs=[], outputs=self.Xg)
        return

    def set_lam_kld(self, lam_kld=1.0):
        """
        Set the relative weight of KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld
        self.lam_kld.set_value(floatX(new_lam))
        return

    def _construct_rvs(self):
        """
        Initialize random values required for generating self.obs_count
        observations from self.gen_network.
        """
        # compute shapes of random values required for the feedforward pass
        # through self.gen_network (to generate self.obs_count observations)
        rand_shapes = self.gen_network.compute_rand_shapes(self.obs_count)
        # initialize random theano shared vars with the appropriate shapes for
        # storing means and log variances to reparametrize gaussian samples
        rv_means = [self.mean_init_func(rs) for rs in rand_shapes]
        rv_logvars = [self.logvar_init_func(rs) for rs in rand_shapes]
        # construct symbolic variables for reparametrized gaussian samples
        rand_vals = []
        for rv_mean, rv_logvar in zip(rv_means, rv_logvars):
            zmuv_gauss = t_rng.normal(size=rv_mean.shape)
            reparam_gauss = rv_mean + (T.exp(0.5*rv_logvar) * zmuv_gauss)
            rand_vals.append(reparam_gauss)
        return rv_means, rv_logvars, rand_vals

    def _construct_nlls(self, x, m, x_hat, out_logvar):
        """
        Compute the reconstruction cost for the ground truth values in x, using
        the reconstructed values in x_hat, ignoring values when m == 0.
        """
        x = T.flatten(x, 2)
        m = T.flatten(m, 2)
        x_hat = T.flatten(x_hat, 2)
        nll = -1.0 * log_prob_gaussian2(x, x_hat, log_vars=out_logvar, mask=m)
        nll = nll.flatten()
        return nll

    def _construct_klds(self):
        """
        Compute KL divergence between reparametrized Gaussians based on
        self.rv_mean/self.rv_logvar, and ZMUV Gaussians.
        """
        all_klds = []
        for rv_mean, rv_logvar in zip(self.rv_means, self.rv_logvars):
            layer_kld = gaussian_kld(mu_left=T.flatten(rv_mean, 2),
                                     logvar_left=T.flatten(rv_logvar, 2),
                                     mu_right=0.0, logvar_right=0.0)
            all_klds.append(T.sum(layer_kld, axis=1))
        obs_klds = sum(all_klds)
        return obs_klds








class DiscNetwork(object):
    """
    A deep convolutional discriminator network. This provides a wrapper around
    a sequence of modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this DiscNetwork. All but
                 the last module should be instances of DiscConvModule. The
                 last module should an instance of DiscFCModule.
    """
    def __init__(self, modules):
        self.modules = [m for m in modules]
        self.fc_module = self.modules[-1]
        self.conv_modules = self.modules[:-1]
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        return

    def apply(self, input, ret_vals=None, app_sigm=True,
              disc_noise=None):
        """
        Apply this DiscNetwork to some input and return some subset of the
        discriminator layer outputs from its underlying modules.
        """
        if ret_vals is None:
            ret_vals = range(len(self.modules))
        hs = [input]
        ys = []
        for i, module in enumerate(self.modules):
            if i == (len(self.modules) - 1):
                # final fc module takes 1d input
                try:
                    hi, yi = module.apply(T.flatten(hs[-1], 2),
                                          noise_sigma=disc_noise)
                except:
                    print("OOPS")
                    print("hs: {}".format(str(hs)))
                    print("disc_noise: {}".format(str(disc_noise)))
                    print("OOPS")
            else:
                # other modules take 2d input
                hi, yi = module.apply(hs[-1], noise_sigma=disc_noise)
            hs.append(hi)
            if i in ret_vals:
                if app_sigm:
                    ys.append(sigmoid(yi))
                else:
                    ys.append(yi)
        return ys
