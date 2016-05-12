from lib import inits
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, deconv, reparametrize, conv_cond_concat, \
    mean_pool_rows, reparametrize_uniform
from lib.theano_utils import floatX, sharedX
from lib.costs import log_prob_bernoulli, log_prob_gaussian, gaussian_kld

#
# Phil's business
#

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()


def tanh_clip(x, bound=10.0):
    """
    Do soft "tanh" clipping to put data in range -scale....+scale.
    """
    x = bound * tanh((1.0 / bound) * x)
    return x


def tanh_clip_softmax(x, bound=10.0):
    """
    Do soft "tanh" clipping to put data in range -scale....+scale.
    """
    x_clip = bound * tanh((1.0 / bound) * x)
    x_clip_sm = T.nnet.softmax(x_clip)
    return x_clip_sm


class GenNetworkGAN(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    sequence of modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this GenNetworkGAN. The
                 first module must be an instance of GenTopModule and the
                 remaining modules should be instances of GenConvModule or
                 BasicConvModule.
        output_transform: transform to apply to output of final convolutional
                          module to get the output of this GenNetworkGAN.
    """
    def __init__(self, modules, output_transform):
        self.modules = [m for m in modules]
        self.fc_module = self.modules[0]
        self.conv_modules = self.modules[1:]
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        self.output_transform = output_transform
        print("Compiling rand shape computer...")
        self.compute_rand_shapes = self._construct_compute_rand_shapes()
        self.rand_shapes = self.compute_rand_shapes(50)
        print("DONE.")
        print("Compiling sample generator...")
        self.generate_samples = self._construct_generate_samples()
        _ = self.generate_samples(50)
        print("DONE.")
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_params(self, f_name=None):
        """
        Load params from a file saved via self.dump_params.
        """
        assert(not (f_name is None))
        pickle_file = open(f_name)
        # reload the parameter dicts for all modules in this network
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.modules):
            mod.load_params(param_dict=param_dict)
        pickle_file.close()
        return

    def apply(self, rand_vals=None, batch_size=None,
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
        result = self.output_transform(acts[-1])
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


class DiscNetworkGAN(object):
    """
    A deep convolutional discriminator network. This provides a wrapper around
    a sequence of modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this DiscNetworkGAN. All but
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

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_params(self, f_name=None):
        """
        Load params from a file saved via self.dump_params.
        """
        assert(not (f_name is None))
        pickle_file = open(f_name)
        # reload the parameter dicts for all modules in this network
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.modules):
            mod.load_params(param_dict=param_dict)
        pickle_file.close()
        return

    def apply(self, input, ret_vals=None, app_sigm=True,
              noise_sigma=None, ret_acts=False, share_mask=False):
        """
        Apply this DiscNetworkGAN to some input and return some subset of the
        discriminator layer outputs from its underlying modules.
        """
        if ret_vals is None:
            ret_vals = range(len(self.modules))
        hs = [input]
        ys = []
        for i, module in enumerate(self.modules):
            if i == (len(self.modules) - 1):
                # final fc module takes 1d input
                result = module.apply(T.flatten(hs[-1], 2),
                                      noise_sigma=noise_sigma,
                                      share_mask=share_mask)
            else:
                # other modules take 2d input
                result = module.apply(hs[-1], noise_sigma=noise_sigma,
                                      share_mask=share_mask)
            if isinstance(result, list):
                hi = result[0]
                yi = result[1]
                if i in ret_vals:
                    if app_sigm:
                        ys.append(T.nnet.sigmoid(yi))
                    else:
                        ys.append(yi)
            else:
                hi = result
            hs.append(hi)
        if ret_acts:
            result = [hs, ys]
        else:
            result = ys
        return result


class VarInfModel(object):
    """
    Wrapper class for extracting variational estimates of log-likelihood from
    a GenNetworkGAN. The VarInfModel will be specific to a given set of inputs.
    """
    def __init__(self, X, M, gen_network, post_logvar=None):
        # observations for which to perform variational inference
        self.X = sharedX(X, name='VIM.X')
        # masks for which components of each observation are "visible"
        self.M = sharedX(M, name='VIM.M')
        self.gen_network = gen_network
        self.post_logvar = post_logvar
        self.obs_count = X.shape[0]
        self.mean_init_func = inits.Normal(loc=0., scale=0.02)
        self.logvar_init_func = inits.Normal(loc=0., scale=0.02)
        # get initial means and log variances of stochastic variables for the
        # observations in self.X, using the GenNetworkGAN self.gen_network. also,
        # make symbolic random variables for passing to self.gen_network.
        self.rv_means, self.rv_logvars, self.rand_vals = self._construct_rvs()
        # get samples from self.gen_network using self.rv_mean/self.rv_logvar
        self.Xg = self.gen_network.apply(rand_vals=self.rand_vals)
        # self.output_logvar modifies the output distribution
        self.output_logvar = sharedX(np.zeros((2,)), name='VIM.output_logvar')
        self.bounded_logvar = 5.0 * T.tanh((1.0 / 5.0) * self.output_logvar[0])
        # compute reconstruction/NLL cost using self.Xg
        self.nlls = self._construct_nlls(x=self.X, m=self.M, x_hat=self.Xg,
                                         out_logvar=self.bounded_logvar)
        # construct symbolic vars for KL divergences between our reparametrized
        # Gaussians, and some ZMUV Gaussians.
        self.lam_kld = sharedX(np.ones((2,)), name='VIM.lam_kld')
        self.set_lam_kld(lam_kld=1.0)
        self.klds = self._construct_klds()
        # make symbolic vars for the overall optimization objective
        self.vfe_bounds = self.nlls + self.klds
        self.opt_cost = T.mean(self.nlls) + (self.lam_kld[0] * T.mean(self.klds))
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
        zero_ary = np.zeros((2,))
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
            zmuv_gauss = cu_rng.normal(size=rv_mean.shape)
            if self.post_logvar is None:
                reparam_gauss = rv_mean + (T.exp(0.5 * rv_logvar) * zmuv_gauss)
            else:
                fixed_logvar = (0.0 * rv_logvar) + self.post_logvar
                reparam_gauss = rv_mean + (T.exp(0.5 * fixed_logvar) * zmuv_gauss)
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
        nll = -1.0 * log_prob_gaussian(x, x_hat, log_vars=out_logvar, mask=m)
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