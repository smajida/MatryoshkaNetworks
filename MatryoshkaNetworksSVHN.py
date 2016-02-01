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
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng
from lib.ops import batchnorm, deconv, reparametrize
from lib.theano_utils import floatX, sharedX
from lib.costs import log_prob_gaussian, gaussian_kld

#
# Phil's business
#

tanh = activations.Tanh()

def tanh_clip(x, bound=10.0):
    """
    Do soft "tanh" clipping to put data in range -scale....+scale.
    """
    x = bound * tanh((1.0 / bound) * x)
    return x


####################################
# Generator network for use in GAN #
####################################

class GenNetworkGAN(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    sequence of modules from MatryoshkaModules.py.

    Params:
        modules: a list of the modules that make up this GenNetworkGAN. The
                 first module must be an instance of GenFCModule and the
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


########################################
# Discriminator network for use in GAN #
########################################

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
              disc_noise=None, ret_acts=False, share_mask=False):
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
                                      noise_sigma=disc_noise,
                                      share_mask=share_mask)
            else:
                # other modules take 2d input
                result = module.apply(hs[-1], noise_sigma=disc_noise,
                                      share_mask=share_mask)
            if type(result) == type([1, 2, 3]):
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


##############################################################
# Model for doing mean-fieldish posterior inference in a GAN #
##############################################################

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
        self.output_logvar = sharedX(np.zeros((1,)), name='VIM.output_logvar')
        self.bounded_logvar = 5.0 * T.tanh((1.0/5.0) * self.output_logvar[0])
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
            zmuv_gauss = cu_rng.normal(size=rv_mean.shape)
            if self.post_logvar is None:
                reparam_gauss = rv_mean + (T.exp(0.5*rv_logvar) * zmuv_gauss)
            else:
                fixed_logvar = (0.0 * rv_logvar) + self.post_logvar
                reparam_gauss = rv_mean + (T.exp(0.5*fixed_logvar) * zmuv_gauss)
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


########################################################################
# Collection of modules for variational inference and generating stuff #
########################################################################

class InfGenModel(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    collection of bottom-up, top-down, and info merging Matryoshka modules.

    Params:
        bu_modules: modules for computing bottom-up (inference) information.
        td_modules: modules for computing top-down (generative) information.
        im_modules: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
        output_transform: transform to apply to outputs of the top-down model.
    """
    def __init__(self,
                 bu_modules, td_modules, im_modules,
                 merge_info, output_transform):
        # grab the bottom-up, top-down, and info merging modules
        self.bu_modules = [m for m in bu_modules]
        self.td_modules = [m for m in td_modules]
        self.im_modules = [m for m in im_modules]
        self.im_modules_dict = {m.mod_name: m for m in im_modules}
        # grab the full set of trainable parameters in these modules
        self.gen_params = []
        self.inf_params = []
        for module in self.td_modules: # top-down is the generator
            self.gen_params.extend(module.params)
        for module in self.bu_modules: # bottom-up is part of inference
            self.inf_params.extend(module.params)
        for module in self.im_modules: # info merge is part of inference
            self.inf_params.extend(module.params)
        # make dist_scale parameter (add it to the inf net parameters)
        self.dist_scale = sharedX( floatX([0.1]) )
        self.inf_params.append(self.dist_scale)
        # store a list of all parameters in this network
        self.params = self.inf_params + self.gen_params
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
        # keep a transform that we'll apply to generator output
        if output_transform == 'ident':
            self.output_transform = lambda x: x
        else:
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

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.bu_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1) # dump BU modules
        mod_param_dicts = [m.dump_params() for m in self.td_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1) # dump TD modules
        mod_param_dicts = [m.dump_params() for m in self.im_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1) # dump IM modules
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
        mod_param_dicts = cPickle.load(pickle_file) # load BU modules
        for param_dict, mod in zip(mod_param_dicts, self.bu_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file) # load TD modules
        for param_dict, mod in zip(mod_param_dicts, self.td_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file) # load IM modules
        for param_dict, mod in zip(mod_param_dicts, self.im_modules):
            mod.load_params(param_dict=param_dict)
        # load dist_scale parameter
        ds_ary = cPickle.load(pickle_file)
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def apply_td(self, rand_vals=None, batch_size=None,
                 rand_shapes=False):
        """
        Apply this generator network using the given random values.
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
        acts = []
        r_shapes = []
        res = None
        for i, rvs in enumerate(rand_vals):
            if i == 0:
                # feedforward through the top-most, fully-connected module
                res = self.td_modules[i].apply(rand_vals=rvs,
                                               batch_size=batch_size,
                                               rand_shapes=rand_shapes)
            else:
                # feedforward through a convolutional module
                res = self.td_modules[i].apply(acts[-1],
                                               rand_vals=rvs,
                                               rand_shapes=rand_shapes)
            if not rand_shapes:
                acts.append(res)
            else:
                acts.append(res[0])
                r_shapes.append(res[1])
        # apply some transform (e.g. tanh or sigmoid) to final activations
        result = self.output_transform(acts[-1])
        if rand_shapes:
            result = r_shapes
        return result

    def apply_bu(self, input):
        """
        Apply this model's bottom-up inference modules to the given input,
        and return a dict mapping BU module names to their outputs.
        """
        bu_acts = []
        res_dict = {}
        for i, bu_mod in enumerate(self.bu_modules):
            if (i == 0):
                res = bu_mod.apply(input)
            else:
                res = bu_mod.apply(bu_acts[i-1])
            bu_acts.append(res)
            res_dict[bu_mod.mod_name] = res
        res_dict['bu_acts'] = bu_acts
        return res_dict

    def apply_im(self, input):
        """
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        This first computes the full bottom-up pass to collect the output of
        each BU module, where the output of the final BU module is the means
        and log variances for a diagonal Gaussian distribution over the latent
        variables that will be fed as input to the first TD module.

        This then computes the top-down pass using latent variables sampled
        from distributions determined by merging partial results of the BU pass
        with results from the partially-completreced TD pass.
        """
        # set aside a dict for recording KLd info at each layer where we use
        # conditional distributions over the latent variables.
        kld_dict = {}
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input)
        # now, run top-down pass using latent variables sampled from
        # conditional distributions constructed by merging bottom-up and
        # top-down information.
        td_acts = []
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            if (td_mod_name in self.merge_info):
                # handle computation for a TD module that requires samples from
                # a conditional distribution formed by merging BU and TD info.
                bu_mod_name = self.merge_info[td_mod_name]['bu_module']
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                if im_mod_name is None:
                    # handle conditionals based purely on BU info
                    cond_mean = bu_res_dict[bu_mod_name][0]
                    cond_logvar = bu_res_dict[bu_mod_name][1]
                    # bound the conditional means if desired
                    cond_mean = tanh_clip(cond_mean, bound=3.0)
                    # bound the conditional logvars if desired
                    cond_logvar = tanh_clip(cond_logvar, bound=3.0)
                    # do reparametrization
                    rand_vals = reparametrize((self.dist_scale[0] * cond_mean),
                                              (self.dist_scale[0] * cond_logvar),
                                              rng=cu_rng)
                    # feedforward through the top-most TD module
                    td_act_i = td_module.apply(rand_vals=rand_vals)
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]              # info from TD pass
                    bu_info = bu_res_dict[bu_mod_name] # info from BU pass
                    im_module = self.im_modules_dict[im_mod_name]
                    cond_mean, cond_logvar = \
                            im_module.apply(td_input=td_info, bu_input=bu_info)
                    # bound the conditional means if desired
                    cond_mean = tanh_clip(cond_mean, bound=3.0)
                    # bound the conditional logvars if desired
                    cond_logvar = tanh_clip(cond_logvar, bound=3.0)
                    rand_vals = reparametrize((self.dist_scale[0] * cond_mean),
                                              (self.dist_scale[0] * cond_logvar),
                                              rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=rand_vals)
                # record TD info produced by current module
                td_acts.append(td_act_i)
                # record KLd info for the relevant conditional distribution
                kld_i = gaussian_kld(T.flatten((self.dist_scale[0] * cond_mean), 2),
                                     T.flatten((self.dist_scale[0] * cond_logvar), 2),
                                     0.0, 0.0)
                kld_dict[td_mod_name] = kld_i
            else:
                # handle computation for a TD module that only requires
                # information from preceding TD modules
                td_info = td_acts[-1] # incoming info from TD pass
                td_act_i = td_module.apply(input=td_info, rand_vals=None)
                td_acts.append(td_act_i)
        td_output = self.output_transform(td_acts[-1])
        im_res_dict = {}
        im_res_dict['td_output'] = td_output
        im_res_dict['kld_dict'] = kld_dict
        im_res_dict['td_acts'] = td_acts
        im_res_dict['bu_acts'] = bu_res_dict['bu_acts']
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

    def _construct_compute_rand_shapes(self):
        """
        Compute the shape of stochastic input for all layers in this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_shapes = self.apply_td(batch_size=batch_size, rand_shapes=True)
        # compile a theano function for computing shapes of the Gaussian latent
        # variables used in the top-down generative model.
        shape_func = theano.function([batch_size], sym_shapes)
        return shape_func