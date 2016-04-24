import cPickle
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
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, deconv, reparametrize, conv_cond_concat, \
    mean_pool_rows
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
        samps = self.generate_samples(50)
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


class InfGenModel(object):
    """
    A deep, hierarchical generator network. This provides a wrapper around a
    collection of bottom-up, top-down, and info-merging Matryoshka modules.

    Params:
        bu_modules: modules for computing bottom-up (inference) information.
        td_modules: modules for computing top-down (generative) information.
        im_modules: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
        output_transform: transform to apply to outputs of the top-down model.
        use_td_noise: whether to apply noise in TD modules
        use_bu_noise: whether to apply noise in BU/IM modules
        train_dist_scale: whether to train rescaling param (for testing)
    """
    def __init__(self,
                 bu_modules, td_modules, im_modules, sc_modules,
                 merge_info, output_transform, use_sc=False,
                 use_td_noise=False,
                 use_bu_noise=True,
                 train_dist_scale=True):
        # grab the bottom-up, top-down, and info merging modules
        self.bu_modules = [m for m in bu_modules]
        self.td_modules = [m for m in td_modules]
        self.im_modules = [m for m in im_modules]
        self.use_sc = use_sc
        if self.use_sc:
            self.sc_modules = [m for m in sc_modules]
        else:
            self.sc_modules = None
        self.im_modules_dict = {m.mod_name: m for m in im_modules}
        self.im_modules_dict[None] = None
        # grab the full set of trainable parameters in these modules
        self.gen_params = []
        self.inf_params = []
        for module in self.td_modules:  # top-down is the generator
            self.gen_params.extend(module.params)
        for module in self.bu_modules:  # bottom-up is part of inference
            self.inf_params.extend(module.params)
        for module in self.im_modules:  # info merge is part of inference
            self.inf_params.extend(module.params)
        if self.use_sc:
            for module in self.sc_modules:  # shortcut is part of inference
                self.inf_params.extend(module.params)
        # filter redundant parameters, to allow parameter sharing
        p_dict = {}
        for p in self.gen_params:
            p_dict[p.name] = p
        self.gen_params = p_dict.values()
        p_dict = {}
        for p in self.inf_params:
            p_dict[p.name] = p
        self.inf_params = p_dict.values()
        # make dist_scale parameter (add it to the inf net parameters)
        if train_dist_scale:
            # init to a somewhat arbitrary value -- not magic (probably)
            self.dist_scale = sharedX(floatX([0.2]))
            self.gen_params.append(self.dist_scale)
        else:
            self.dist_scale = sharedX(floatX([1.0]))
        # gather a list of all parameters in this network
        self.params = self.inf_params + self.gen_params
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
        # keep a transform that we'll apply to generator output
        if output_transform == 'ident':
            self.output_transform = lambda x: x
        else:
            self.output_transform = output_transform
        # derp derp
        self.use_td_noise = use_td_noise
        self.use_bu_noise = use_bu_noise
        print("Compiling sample generator...")
        # i'm the operator with my sample generator
        self.generate_samples = self._construct_generate_samples()
        samps = self.generate_samples(32)
        print("DONE.")
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.bu_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump BU modules
        mod_param_dicts = [m.dump_params() for m in self.td_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump TD modules
        mod_param_dicts = [m.dump_params() for m in self.im_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump IM modules
        if self.use_sc:
            mod_param_dicts = [m.dump_params() for m in self.sc_modules]
            cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump SC modules
        # dump dist_scale parameter
        ds_ary = self.dist_scale.get_value(borrow=False)
        cPickle.dump(ds_ary, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_params(self, f_name=None):
        """
        Load params from a file saved via self.dump_params.
        """
        assert(not (f_name is None))
        pickle_file = open(f_name)
        # reload the parameter dicts for all modules in this network
        mod_param_dicts = cPickle.load(pickle_file)  # load BU modules
        for param_dict, mod in zip(mod_param_dicts, self.bu_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)  # load TD modules
        for param_dict, mod in zip(mod_param_dicts, self.td_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)  # load IM modules
        for param_dict, mod in zip(mod_param_dicts, self.im_modules):
            mod.load_params(param_dict=param_dict)
        if self.use_sc:
            mod_param_dicts = cPickle.load(pickle_file)  # load SC modules
            for param_dict, mod in zip(mod_param_dicts, self.sc_modules):
                mod.load_params(param_dict=param_dict)
        # load dist_scale parameter
        ds_ary = cPickle.load(pickle_file)
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def apply_td(self, rand_vals=None, batch_size=None, noise=None):
        """
        Compute a stochastic top-down pass using the given random values.
        -- batch_size must be provided if rand_vals is None, so we can
           determine the appropriate size for latent samples.
        """
        assert not ((batch_size is None) and (rand_vals is None)), \
            "need _either_ batch_size or rand_vals."
        assert ((batch_size is None) or (rand_vals is None)), \
            "need _either_ batch_size or rand_vals."
        assert ((rand_vals is None) or (len(rand_vals) == len(self.td_modules))), \
            "random values should be appropriate for this network."
        if rand_vals is None:
            # no random values were provided, which means we'll be generating
            # based on a user-provided batch_size.
            rand_vals = [None for i in range(len(self.td_modules))]
        td_noise = noise if self.use_td_noise else None
        td_acts = []
        for i, (rvs, td_module) in enumerate(zip(rand_vals, self.td_modules)):
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            im_module = self.im_modules_dict[im_mod_name]
            if td_mod_type in ['top', 'cond']:
                # handle computation for a TD module that requires
                # sampling some stochastic latent variables.
                if td_mod_type == 'top':
                    # feedforward through the top-most generator module.
                    # this module has a fixed ZMUV Gaussian prior.
                    td_act_i = td_module.apply(rand_vals=rvs,
                                               batch_size=batch_size,
                                               noise=td_noise)
                else:
                    # feedforward through an internal TD module
                    cond_mean_td, cond_logvar_td = \
                        im_module.apply_td(td_input=td_acts[-1],
                                           noise=td_noise)
                    cond_mean_td = self.dist_scale[0] * cond_mean_td
                    cond_logvar_td = self.dist_scale[0] * cond_logvar_td
                    if rvs is None:
                        rvs = reparametrize(cond_mean_td, cond_logvar_td,
                                            rng=cu_rng)
                    else:
                        rvs = reparametrize(cond_mean_td, cond_logvar_td,
                                            rvs=rvs)
                    # feedforward using the reparametrized latent variable
                    # samples and incoming activations.
                    td_act_i = td_module.apply(input=td_acts[-1],
                                               rand_vals=rvs,
                                               noise=td_noise)
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules (no rand input)
                td_act_i = td_module.apply(input=td_acts[-1], rand_vals=None,
                                           noise=td_noise)
            td_acts.append(td_act_i)
        # apply some transform (e.g. tanh or sigmoid) to final activations
        result = self.output_transform(td_acts[-1])
        return result

    def apply_bu(self, input, noise=None, sc_info=None):
        """
        Apply this model's bottom-up inference modules to the given input,
        and return a dict mapping BU module names to their outputs.
        """
        bu_noise = noise if self.use_bu_noise else None
        bu_acts = []
        res_dict = {}
        for i, bu_mod in enumerate(self.bu_modules):
            if (i == 0):
                bu_info = input
            else:
                bu_info = bu_acts[i - 1]
            if self.use_sc and (i == (len(self.bu_modules) - 1)):
                # add shortcut info for the top-most inference module
                bu_info = T.concatenate([bu_info, sc_info], axis=1)
            res = bu_mod.apply(bu_info, noise=bu_noise)
            bu_acts.append(res)
            res_dict[bu_mod.mod_name] = res
        res_dict['bu_acts'] = bu_acts
        return res_dict

    def apply_sc(self, input, noise=None):
        """
        Apply this model's shortcut inference modules to the given input,
        and return the output of final shortcut module.
        """
        if self.use_sc:
            bu_noise = noise if self.use_bu_noise else None
            sc_acts = []
            for i, sc_mod in enumerate(self.sc_modules):
                if (i == 0):
                    res = sc_mod.apply(input, noise=bu_noise)
                else:
                    res = sc_mod.apply(sc_acts[i - 1], noise=bu_noise)
                sc_acts.append(res)
            sc_info = sc_acts[-1]
        else:
            sc_info = None
        return sc_info

    def apply_im(self, input, noise=None):
        """
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        -- this does all the heavy lifting --

        This first computes the full bottom-up pass to collect the output of
        each BU module, where the output of the final BU module is the means
        and log variances for a diagonal Gaussian distribution over the latent
        variables that will be fed as input to the first TD module.

        This then computes the top-down pass using latent variables sampled
        from distributions determined by merging partial results of the BU pass
        with results from the partially-completed TD pass. The IM modules feed
        information to eachother too.
        """
        bu_noise = noise if self.use_bu_noise else None
        td_noise = noise if self.use_td_noise else None
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        kld_dict = {}
        z_dict = {}
        logz_dict = {'log_p_z': [], 'log_q_z': []}
        # first, run the bottom-up pass
        sc_info = self.apply_sc(input=input, noise=bu_noise)
        bu_res_dict = self.apply_bu(input=input, noise=bu_noise,
                                    sc_info=sc_info)
        # dict for storing IM state information
        im_res_dict = {None: None}
        # now, run top-down pass using latent variables sampled from
        # conditional distributions constructed by merging bottom-up and
        # top-down information.
        td_acts = []
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            bu_src_name = self.merge_info[td_mod_name]['bu_source']
            im_src_name = self.merge_info[td_mod_name]['im_source']
            im_module = self.im_modules_dict[im_mod_name]  # this might be None
            if td_mod_type in ['top', 'cond']:
                if td_mod_type == 'top':
                    # top TD conditionals are based purely on BU info
                    cond_mean_im = bu_res_dict[bu_src_name][0]
                    cond_logvar_im = bu_res_dict[bu_src_name][1]
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    cond_mean_td = 0.0 * cond_mean_im
                    cond_logvar_td = 0.0 * cond_logvar_im
                    # reparametrize Gaussian for latent samples
                    cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                             rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(rand_vals=cond_rvs,
                                               noise=td_noise)
                    # compute initial state for IM pass, maybe...
                    im_act_i = None
                    if not (im_module is None):
                        im_act_i = im_module.apply(rand_vals=cond_rvs,
                                                   noise=bu_noise)
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]               # info from TD pass
                    bu_info = bu_res_dict[bu_src_name]  # info from BU pass
                    im_info = im_res_dict[im_src_name]  # info from IM pass
                    if self.use_sc:
                        # add shortcut info to bottom-up info
                        bu_info = T.concatenate([bu_info, sc_info], axis=1)
                    # get the conditional distribution SSs (Sufficient Stat s)
                    cond_mean_im, cond_logvar_im, im_act_i = \
                        im_module.apply_im(td_input=td_info,
                                           bu_input=bu_info,
                                           im_input=im_info,
                                           noise=bu_noise)
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    cond_mean_td, cond_logvar_td = \
                        im_module.apply_td(td_input=td_info,
                                           noise=td_noise)
                    cond_mean_td = self.dist_scale[0] * cond_mean_td
                    cond_logvar_td = self.dist_scale[0] * cond_logvar_td
                    # estimate location as an offset from prior
                    cond_mean_im = cond_mean_im + cond_mean_td

                    # reparametrize Gaussian for latent samples
                    cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                             rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=cond_rvs,
                                               noise=td_noise)
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict[im_mod_name] = im_act_i
                # record KLd info for the conditional distribution
                kld_i = gaussian_kld(T.flatten(cond_mean_im, 2),
                                     T.flatten(cond_logvar_im, 2),
                                     0.0, 0.0)
                kld_dict[td_mod_name] = kld_i
                # get the log likelihood of the current latent samples under
                # both the proposal distribution q(z | x) and the prior p(z).
                # -- these are used when computing the IWAE bound.
                log_p_z = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                            T.flatten(cond_mean_td, 2),
                                            log_vars=T.flatten(cond_logvar_td, 2),
                                            do_sum=True)
                log_q_z = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                            T.flatten(cond_mean_im, 2),
                                            log_vars=T.flatten(cond_logvar_im, 2),
                                            do_sum=True)
                logz_dict['log_p_z'].append(log_p_z)
                logz_dict['log_q_z'].append(log_q_z)
                z_dict[td_mod_name] = cond_rvs
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules
                td_info = td_acts[-1]  # incoming info from TD pass
                td_act_i = td_module.apply(input=td_info, rand_vals=None,
                                           noise=td_noise)
                td_acts.append(td_act_i)
                if not (im_module is None):
                    # perform an update of the IM state
                    im_info = im_res_dict[im_src_name]
                    im_act_i = im_module.apply(input=im_info, rand_vals=None,
                                               noise=bu_noise)
                    im_res_dict[im_mod_name] = im_act_i
            else:
                assert False, "BAD td_mod_type: {}".format(td_mod_type)
        # apply output transform (into observation space, presumably), to get
        # the final "reconstruction" produced by the merged BU/TD pass.
        td_output = self.output_transform(td_acts[-1])
        # package results into a handy dictionary
        im_res_dict = {}
        im_res_dict['td_output'] = td_output
        im_res_dict['kld_dict'] = kld_dict
        im_res_dict['td_acts'] = td_acts
        im_res_dict['bu_acts'] = bu_res_dict['bu_acts']
        im_res_dict['z_dict'] = z_dict
        im_res_dict['log_p_z'] = logz_dict['log_p_z']
        im_res_dict['log_q_z'] = logz_dict['log_q_z']
        return im_res_dict

    def _construct_generate_samples(self):
        """
        Generate some samples from this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_samples = self.apply_td(batch_size=batch_size)
        # compile a theano function for sampling outputs from the top-down
        # generative model.
        sample_func = theano.function([batch_size], sym_samples)
        return sample_func


class SimpleMLP(object):
    """
    A simple feedforward network. This wraps a sequence of fully connected
    modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this SimpleMLP.
    """
    def __init__(self, modules=None):
        assert not (modules is None), "Don't be a dunce! Supply modules!"
        self.modules = [m for m in modules]
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        mod_param_dicts = [m.dump_params() for m in self.modules]
        if not (f_name is None):
            # dump param dict to the given file
            f_handle = file(f_name, 'wb')
            cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
            f_handle.close()
        return mod_param_dicts

    def load_params(self, f_name=None, mod_param_dicts=None):
        """
        Load params from a file saved via self.dump_params.
        """
        # reload the parameter dicts for all modules in this network
        if not (f_name is None):
            # reload params from a file
            pickle_file = open(f_name)
            mod_param_dicts = cPickle.load(pickle_file)
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
            pickle_file.close()
        else:
            # reload params from a dict
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
        return

    def apply(self, input, noise=None):
        """
        Apply this SimpleMLP to some input and return the output of
        its final layer.
        """
        hs = [input]
        for i, module in enumerate(self.modules):
            hi = module.apply(T.flatten(hs[-1], 2), noise=noise)
            hs.append(hi)
        return hs[-1]


class CondInfGenModel(object):
    '''
    A deep, hierarchical conditional generator network. This provides a wrapper
    around a collection of bottom-up, top-down, and info-merging Matryoshka
    modules.

    Params:
        td_modules: modules for computing top-down (generative) information.
                    -- these are all shared between generator and inferencer
        bu_modules_gen: modules for computing bottom-up information.
        im_modules_gen: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        bu_modules_inf: modules for computing bottom-up information.
        im_modules_inf: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
        output_transform: transform to apply to outputs of the top-down model.
        train_dist_scale: whether to train rescaling param (for testing)
    '''
    def __init__(self,
                 td_modules,
                 bu_modules_gen, im_modules_gen,
                 bu_modules_inf, im_modules_inf,
                 merge_info, output_transform,
                 train_dist_scale=True):
        # organize the always shared modules
        self.td_modules = [m for m in td_modules]
        # organize the modules (maybe) specific to the generator
        self.bu_modules_gen = [m for m in bu_modules_gen]
        self.im_modules_gen = [m for m in im_modules_gen]
        self.im_modules_gen_dict = {m.mod_name: m for m in im_modules_gen}
        self.im_modules_gen_dict[None] = None
        # organize the modules (maybe) specific to the inferencer
        self.bu_modules_inf = [m for m in bu_modules_inf]
        self.im_modules_inf = [m for m in im_modules_inf]
        self.im_modules_inf_dict = {m.mod_name: m for m in im_modules_inf}
        self.im_modules_inf_dict[None] = None
        # grab the full set of trainable parameters in these modules
        self.gen_params = []
        self.inf_params = []
        for module in self.td_modules:  # top-down is the generator
            self.gen_params.extend(module.params)
        for module in self.bu_modules_inf:  # bottom-up is part of inference
            self.inf_params.extend(module.params)
        for module in self.im_modules_inf:  # info merge is part of inference
            self.inf_params.extend(module.params)
        # filter redundant parameters, to allow parameter sharing
        p_dict = {}
        for p in self.gen_params:
            p_dict[p.name] = p
        self.gen_params = p_dict.values()
        p_dict = {}
        for p in self.inf_params:
            p_dict[p.name] = p
        self.inf_params = p_dict.values()
        # make dist_scale parameter (add it to the inf net parameters)
        if train_dist_scale:
            # init to a somewhat arbitrary value -- not magic (probably)
            self.dist_scale = sharedX(floatX([0.2]))
            self.gen_params.append(self.dist_scale)
        else:
            self.dist_scale = sharedX(floatX([1.0]))
        # gather a list of all parameters in this network
        self.params = self.inf_params + self.gen_params
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
        # keep a transform that we'll apply to generator output
        if output_transform == 'ident':
            self.output_transform = lambda x: x
        else:
            self.output_transform = output_transform
        print("Compiling sample generator...")
        # i'm the operator with my sample generator
        self.generate_samples = self._construct_generate_samples()
        samps = self.generate_samples(32)
        print("DONE.")
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.bu_modules_inf]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump BU modules
        mod_param_dicts = [m.dump_params() for m in self.td_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump TD modules
        mod_param_dicts = [m.dump_params() for m in self.im_modules_inf]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)  # dump IM modules
        # dump dist_scale parameter
        ds_ary = self.dist_scale.get_value(borrow=False)
        cPickle.dump(ds_ary, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_params(self, f_name=None):
        """
        Load params from a file saved via self.dump_params.
        """
        assert(not (f_name is None))
        pickle_file = open(f_name)
        # reload the parameter dicts for all modules in this network
        mod_param_dicts = cPickle.load(pickle_file)  # load BU modules
        for param_dict, mod in zip(mod_param_dicts, self.bu_modules_inf):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)  # load TD modules
        for param_dict, mod in zip(mod_param_dicts, self.td_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)  # load IM modules
        for param_dict, mod in zip(mod_param_dicts, self.im_modules_inf):
            mod.load_params(param_dict=param_dict)
        # load dist_scale parameter
        ds_ary = cPickle.load(pickle_file)
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def apply_td(self, rand_vals=None, batch_size=None):
        '''
        Compute a stochastic top-down pass using the given random values.
        -- batch_size must be provided if rand_vals is None, so we can
           determine the appropriate size for latent samples.
        '''
        assert not ((batch_size is None) and (rand_vals is None)), \
            "need _either_ batch_size or rand_vals."
        assert ((batch_size is None) or (rand_vals is None)), \
            "need _either_ batch_size or rand_vals."
        assert ((rand_vals is None) or (len(rand_vals) == len(self.td_modules))), \
            "random values should be appropriate for this network."
        if rand_vals is None:
            # no random values were provided, which means we'll be generating
            # based on a user-provided batch_size.
            rand_vals = [None for i in range(len(self.td_modules))]
        td_acts = []
        for i, (rvs, td_module) in enumerate(zip(rand_vals, self.td_modules)):
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            im_module = self.im_modules_inf_dict[im_mod_name]
            if td_mod_type in ['top', 'cond']:
                # handle computation for a TD module that requires
                # sampling some stochastic latent variables.
                if td_mod_type == 'top':
                    # feedforward through the top-most generator module.
                    # this module has a fixed ZMUV Gaussian prior.
                    td_act_i = td_module.apply(rand_vals=rvs,
                                               batch_size=batch_size)
                else:
                    # feedforward through an internal TD module
                    cond_mean_td, cond_logvar_td = \
                        im_module.apply_td(td_input=td_acts[-1])
                    cond_mean_td = self.dist_scale[0] * cond_mean_td
                    cond_logvar_td = self.dist_scale[0] * cond_logvar_td
                    if rvs is None:
                        rvs = reparametrize(cond_mean_td, cond_logvar_td,
                                            rng=cu_rng)
                    else:
                        rvs = reparametrize(cond_mean_td, cond_logvar_td,
                                            rvs=rvs)
                    # feedforward using the reparametrized latent variable
                    # samples and incoming activations.
                    td_act_i = td_module.apply(input=td_acts[-1],
                                               rand_vals=rvs)
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules (no rand input)
                td_act_i = td_module.apply(input=td_acts[-1], rand_vals=None)
            td_acts.append(td_act_i)
        # apply some transform (e.g. tanh or sigmoid) to final activations
        result = self.output_transform(td_acts[-1])
        return result

    def apply_bu(self, input):
        '''
        Apply this model's bottom-up inference modules to the given input,
        and return a dict mapping BU module names to their outputs.
        '''
        bu_acts = []
        res_dict = {}
        for i, bu_mod in enumerate(self.bu_modules_inf):
            if (i == 0):
                bu_info = input
            else:
                bu_info = bu_acts[i - 1]
            res = bu_mod.apply(bu_info)
            bu_acts.append(res)
            res_dict[bu_mod.mod_name] = res
        res_dict['bu_acts'] = bu_acts
        return res_dict

    def apply_im(self, input):
        '''
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        -- this does all the heavy lifting --

        This first computes the full bottom-up pass to collect the output of
        each BU module, where the output of the final BU module is the means
        and log variances for a diagonal Gaussian distribution over the latent
        variables that will be fed as input to the first TD module.

        This then computes the top-down pass using latent variables sampled
        from distributions determined by merging partial results of the BU pass
        with results from the partially-completed TD pass. The IM modules feed
        information to eachother too.
        '''
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        kld_dict = {}
        z_dict = {}
        logz_dict = {'log_p_z': [], 'log_q_z': []}
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input=input)
        # dict for storing IM state information
        im_res_dict = {None: None}
        # now, run top-down pass using latent variables sampled from
        # conditional distributions constructed by merging bottom-up and
        # top-down information.
        td_acts = []
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            bu_src_name = self.merge_info[td_mod_name]['bu_source']
            im_src_name = self.merge_info[td_mod_name]['im_source']
            im_module = self.im_modules_inf_dict[im_mod_name]  # this might be None
            if td_mod_type in ['top', 'cond']:
                if td_mod_type == 'top':
                    # top TD conditionals are based purely on BU info
                    cond_mean_im = bu_res_dict[bu_src_name][0]
                    cond_logvar_im = bu_res_dict[bu_src_name][1]
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    cond_mean_td = 0.0 * cond_mean_im
                    cond_logvar_td = 0.0 * cond_logvar_im
                    # reparametrize Gaussian for latent samples
                    cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                             rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(rand_vals=cond_rvs)
                    # compute initial state for IM pass, maybe...
                    im_act_i = None
                    if not (im_module is None):
                        im_act_i = im_module.apply(rand_vals=cond_rvs)
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]               # info from TD pass
                    bu_info = bu_res_dict[bu_src_name]  # info from BU pass
                    im_info = im_res_dict[im_src_name]  # info from IM pass
                    if self.use_sc:
                        # add shortcut info to bottom-up info
                        bu_info = T.concatenate([bu_info, sc_info], axis=1)
                    # get the conditional distribution SSs (Sufficient Stat s)
                    cond_mean_im, cond_logvar_im, im_act_i = \
                        im_module.apply_im(td_input=td_info,
                                           bu_input=bu_info,
                                           im_input=im_info)
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    cond_mean_td, cond_logvar_td = \
                        im_module.apply_td(td_input=td_info)
                    cond_mean_td = self.dist_scale[0] * cond_mean_td
                    cond_logvar_td = self.dist_scale[0] * cond_logvar_td
                    # estimate location as an offset from prior
                    cond_mean_im = cond_mean_im + cond_mean_td

                    # reparametrize Gaussian for latent samples
                    cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                             rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=cond_rvs)
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict[im_mod_name] = im_act_i
                # record KLd info for the conditional distribution
                kld_i = gaussian_kld(T.flatten(cond_mean_im, 2),
                                     T.flatten(cond_logvar_im, 2),
                                     0.0, 0.0)
                kld_dict[td_mod_name] = kld_i
                # get the log likelihood of the current latent samples under
                # both the proposal distribution q(z | x) and the prior p(z).
                # -- these are used when computing the IWAE bound.
                log_p_z = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                            T.flatten(cond_mean_td, 2),
                                            log_vars=T.flatten(cond_logvar_td, 2),
                                            do_sum=True)
                log_q_z = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                            T.flatten(cond_mean_im, 2),
                                            log_vars=T.flatten(cond_logvar_im, 2),
                                            do_sum=True)
                logz_dict['log_p_z'].append(log_p_z)
                logz_dict['log_q_z'].append(log_q_z)
                z_dict[td_mod_name] = cond_rvs
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules
                td_info = td_acts[-1]  # incoming info from TD pass
                td_act_i = td_module.apply(input=td_info, rand_vals=None)
                td_acts.append(td_act_i)
                if not (im_module is None):
                    # perform an update of the IM state
                    im_info = im_res_dict[im_src_name]
                    im_act_i = im_module.apply(input=im_info, rand_vals=None)
                    im_res_dict[im_mod_name] = im_act_i
            else:
                assert False, "BAD td_mod_type: {}".format(td_mod_type)
        # apply output transform (into observation space, presumably), to get
        # the final "reconstruction" produced by the merged BU/TD pass.
        td_output = self.output_transform(td_acts[-1])
        # package results into a handy dictionary
        im_res_dict = {}
        im_res_dict['td_output'] = td_output
        im_res_dict['kld_dict'] = kld_dict
        im_res_dict['td_acts'] = td_acts
        im_res_dict['bu_acts'] = bu_res_dict['bu_acts']
        im_res_dict['z_dict'] = z_dict
        im_res_dict['log_p_z'] = logz_dict['log_p_z']
        im_res_dict['log_q_z'] = logz_dict['log_q_z']
        return im_res_dict

    def _construct_generate_samples(self):
        """
        Generate some samples from this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_samples = self.apply_td(batch_size=batch_size)
        # compile a theano function for sampling outputs from the top-down
        # generative model.
        sample_func = theano.function([batch_size], sym_samples)
        return sample_func


class SimpleMLP(object):
    """
    A simple feedforward network. This wraps a sequence of fully connected
    modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this SimpleMLP.
    """
    def __init__(self, modules=None):
        assert not (modules is None), "Don't be a dunce! Supply modules!"
        self.modules = [m for m in modules]
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        mod_param_dicts = [m.dump_params() for m in self.modules]
        if not (f_name is None):
            # dump param dict to the given file
            f_handle = file(f_name, 'wb')
            cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
            f_handle.close()
        return mod_param_dicts

    def load_params(self, f_name=None, mod_param_dicts=None):
        """
        Load params from a file saved via self.dump_params.
        """
        # reload the parameter dicts for all modules in this network
        if not (f_name is None):
            # reload params from a file
            pickle_file = open(f_name)
            mod_param_dicts = cPickle.load(pickle_file)
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
            pickle_file.close()
        else:
            # reload params from a dict
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
        return

    def apply(self, input, noise=None):
        """
        Apply this SimpleMLP to some input and return the output of
        its final layer.
        """
        hs = [input]
        for i, module in enumerate(self.modules):
            hi = module.apply(T.flatten(hs[-1], 2), noise=noise)
            hs.append(hi)
        return hs[-1]



class SimpleInfMLP(object):
    """
    A simple feedforward network. This wraps a sequence of fully connected
    modules from MatryoshkaModules.py. Assume the final module is InfTopModule.

    Params:
        modules: a list of the modules that make up this SimpleMLP.
    """
    def __init__(self, modules=None):
        assert not (modules is None), "Don't be a dunce! Supply modules!"
        self.modules = [m for m in modules]
        self.params = []
        for module in self.modules:
            self.params.extend(module.params)
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        mod_param_dicts = [m.dump_params() for m in self.modules]
        if not (f_name is None):
            # dump param dict to the given file
            f_handle = file(f_name, 'wb')
            cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
            f_handle.close()
        return mod_param_dicts

    def load_params(self, f_name=None, mod_param_dicts=None):
        """
        Load params from a file saved via self.dump_params.
        """
        # reload the parameter dicts for all modules in this network
        if not (f_name is None):
            # reload params from a file
            pickle_file = open(f_name)
            mod_param_dicts = cPickle.load(pickle_file)
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
            pickle_file.close()
        else:
            # reload params from a dict
            for param_dict, mod in zip(mod_param_dicts, self.modules):
                mod.load_params(param_dict=param_dict)
        return

    def apply(self, input, noise=None):
        """
        Apply this SimpleMLP to some input and return the output of
        its final layer.
        """
        hs = [input]
        for i, module in enumerate(self.modules):
            hi = module.apply(T.flatten(hs[-1], 2), noise=noise)
            hs.append(hi)
        return hs[-1]














##############
# EYE BUFFER #
##############
