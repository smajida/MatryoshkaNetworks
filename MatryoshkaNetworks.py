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
from lib.ops import batchnorm, deconv, reparametrize, conv_cond_concat
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
        self.output_logvar = sharedX(np.zeros((2,)), name='VIM.output_logvar')
        self.bounded_logvar = 5.0 * T.tanh((1.0/5.0) * self.output_logvar[0])
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
        self.dist_scale = sharedX( floatX([0.1,0.1]) )
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
        print("Compiling rand shape computer...")
        self.compute_rand_shapes = self._construct_compute_rand_shapes()
        self.rand_shapes = self.compute_rand_shapes(32)
        print("DONE.")
        print("Compiling sample generator...")
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
        ds_ary = np.zeros((2,)) + ds_ary[0]
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def infer_rand_shapes(self, batch_size):
        """
        Helper function for inferring rand val shapes for gen layers.
        """
        acts = []
        r_shapes = []
        for i, td_module in enumerate(self.td_modules):
            if i == 0:
                # feedforward through the top-most, fully-connected module
                res = td_module.apply(rand_vals=None,
                                      batch_size=batch_size,
                                      rand_shapes=True)
            else:
                # feedforward through a convolutional module
                res = td_module.apply(acts[-1],
                                      rand_vals=None,
                                      rand_shapes=True)
            acts.append(res[0])
            r_shapes.append(res[1])
        return r_shapes

    def apply_td(self, rand_vals=None, batch_size=None):
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
        td_acts = []
        for rvs, td_module, rvs_shape in zip(rand_vals, self.td_modules, self.rand_shapes):
            td_mod_name = td_module.mod_name
            td_act_i = None # this will be set to the output of td_module
            if td_mod_name in self.merge_info:
                # handle computation for a TD module that requires
                # sampling some stochastic latent variables.
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                if im_mod_name is None:
                    # feedforward through the top-most generator module.
                    # this module has a fixed ZMUV Gaussian prior.
                    td_act_i = td_module.apply(rand_vals=rvs,
                                               batch_size=batch_size)
                else:
                    # feedforward through a convolutional module
                    im_module = self.im_modules_dict[im_mod_name]
                    if rvs is None:
                        # sample values to reparametrize, if none given
                        b_size = td_acts[-1].shape[0]
                        rvs_size = (b_size, rvs_shape[1], rvs_shape[2], rvs_shape[3])
                        rvs = cu_rng.normal(size=rvs_size, dtype=theano.config.floatX)
                    if im_module.use_td_cond:
                        # use top-down conditioning
                        cond_mean_td, cond_logvar_td = \
                                im_module.apply_td(td_acts[-1])
                        cond_rvs = reparametrize(cond_mean_td,
                                                 cond_logvar_td,
                                                 rvs=rvs)
                    else:
                        # use samples without reparametrizing
                        cond_rvs = rvs
                    # feedforward using the reparametrized stochastic
                    # variables and incoming activations.
                    td_act_i = td_module.apply(td_acts[-1],
                                               rand_vals=cond_rvs)
            else:
                # handle computation for a TD module that only requires
                # information from preceding TD modules (no rand input)
                td_act_i = td_module.apply(input=td_acts[-1], rand_vals=None)
            td_acts.append(td_act_i)
        # apply some transform (e.g. tanh or sigmoid) to final activations
        result = self.output_transform(td_acts[-1])
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
        z_dict = {}
        logz_dict = {'log_p_z': [], 'log_q_z': []}
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
                    cond_mean_im = bu_res_dict[bu_mod_name][0]
                    cond_logvar_im = bu_res_dict[bu_mod_name][1]
                    cond_mean_im = self.dist_scale[0] * tanh_clip(cond_mean_im, bound=3.0)
                    cond_logvar_im = self.dist_scale[0] * tanh_clip(cond_logvar_im, bound=3.0)
                    # get top-down mean and logvar (here, just ZMUV)
                    cond_mean_td = 0.0 * cond_mean_im
                    cond_logvar_td = 0.0 * cond_logvar_im
                    # do reparametrization
                    rand_vals = reparametrize(cond_mean_im, cond_logvar_im,
                                              rng=cu_rng)
                    # feedforward through the top-most TD module
                    td_act_i = td_module.apply(rand_vals=rand_vals)
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]              # info from TD pass
                    bu_info = bu_res_dict[bu_mod_name] # info from BU pass
                    im_module = self.im_modules_dict[im_mod_name]
                    # get the inference distribution
                    cond_mean_im, cond_logvar_im = \
                            im_module.apply_im(td_input=td_info, bu_input=bu_info)
                    cond_mean_im = self.dist_scale[0] * tanh_clip(cond_mean_im, bound=3.0)
                    cond_logvar_im = self.dist_scale[0] * tanh_clip(cond_logvar_im, bound=3.0)
                    # get the model distribution
                    if im_module.use_td_cond:
                        # get the top-down conditional distribution
                        cond_mean_td, cond_logvar_td = \
                                im_module.apply_td(td_info)
                    else:
                        # use a fixed ZMUV Gaussian prior
                        cond_mean_td = 0.0 * cond_mean_im
                        cond_logvar_td = 0.0 * cond_logvar_im
                    # reparametrize
                    rand_vals = reparametrize(cond_mean_im, cond_logvar_im,
                                              rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=rand_vals)
                # record log probability of z under p and q, for IWAE bound
                log_p_z = log_prob_gaussian(T.flatten(rand_vals, 2),
                                            T.flatten(cond_mean_td, 2),
                                            log_vars=T.flatten(cond_logvar_td, 2),
                                            do_sum=True)
                log_q_z = log_prob_gaussian(T.flatten(rand_vals, 2),
                                            T.flatten(cond_mean_im, 2),
                                            log_vars=T.flatten(cond_logvar_im, 2),
                                            do_sum=True)
                logz_dict['log_p_z'].append(log_p_z)
                logz_dict['log_q_z'].append(log_q_z)
                z_dict[td_mod_name] = rand_vals
                # record TD info produced by current module
                td_acts.append(td_act_i)
                # record KLd info for the relevant conditional distribution
                kld_i = gaussian_kld(T.flatten(cond_mean_im, 2),
                                     T.flatten(cond_logvar_im, 2),
                                     T.flatten(cond_mean_td, 2),
                                     T.flatten(cond_logvar_td, 2))
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

    def _construct_compute_rand_shapes(self):
        """
        Compute the shape of stochastic input for all layers in this network.
        """
        batch_size = T.lscalar()
        # feedforward through the model with batch size "batch_size"
        sym_shapes = self.infer_rand_shapes(batch_size)
        # compile a theano function for computing shapes of the Gaussian latent
        # variables used in the top-down generative model.
        shape_func = theano.function([batch_size], sym_shapes)
        return shape_func

##########################################
# SIMPLEST MLP EVER -- NO TIME TO WASTE! #
##########################################

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

    def apply(self, input):
        """
        Apply this SimpleMLP to some input and return the output of
        its final layer.
        """
        hs = [input]
        for i, module in enumerate(self.modules):
            hi = module.apply(T.flatten(hs[-1], 2))
            hs.append(hi)
        return hs[-1]


class SimpleInfMLP(object):
    """
    A simple feedforward network. This wraps a sequence of fully connected
    modules from MatryoshkaModules.py. Assume the final module is InfFCModule.

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

    def apply(self, input):
        """
        Apply this SimpleMLP to some input and return the output of
        its final layer.
        """
        hs = [input]
        for i, module in enumerate(self.modules):
            hi = module.apply(T.flatten(hs[-1], 2))
            hs.append(hi)
        return hs[-1]

########################################################################
# Collection of modules for variational inference and generating stuff #
########################################################################

class InfGenModelSS(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    collection of bottom-up, top-down, and info merging Matryoshka modules.

    Sampling in this model is conditioned on a "class indicator", throughout
    the top-down generative process.

    Params:
        nyc: number of classes (i.e. indicator dim)
        nbatch: force a fixed batch size for "marginalizing" top-down passes
                --- this is lazy, but it'll do for now (allows code reuse)
        q_aIx_model:   SimpleInfMLP for q(a | x)
        q_yIax_model:  SimpleInfMLP for q(y | a, x)
        q_z0Iyx_model: SimpleInfMLP for q(z0 | y, x)
        bu_modules: modules for computing bottom-up (inference) information.
        td_modules: modules for computing top-down (generative) information.
        im_modules: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
        output_transform: transform to apply to outputs of the top-down model.
    """
    def __init__(self, nyc, nbatch,
                 q_aIx_model, q_yIax_model, q_z0Iyx_model,
                 bu_modules, td_modules, im_modules,
                 merge_info, output_transform):
        # get indicator dimension and top-down batch size
        self.nyc = nyc
        self.nbatch = nbatch
        self.log_nyc = np.log(self.nyc).astype(theano.config.floatX)
        # get models for top-most inference business
        self.q_aIx_model = q_aIx_model
        self.q_yIax_model = q_yIax_model
        self.q_z0Iyx_model = q_z0Iyx_model
        # grab the bottom-up, top-down, and info merging modules
        self.bu_modules = [m for m in bu_modules]
        self.td_modules = [m for m in td_modules]
        self.im_modules = [m for m in im_modules]
        self.im_modules_dict = {m.mod_name: m for m in im_modules}
        # get parameters from top-most inference models
        self.all_params = []
        self.all_params.extend(self.q_aIx_model.params)
        self.all_params.extend(self.q_yIax_model.params)
        self.all_params.extend(self.q_z0Iyx_model.params)
        # get parameters from module lists
        for module in self.td_modules:
            self.all_params.extend(module.params)
        for module in self.bu_modules:
            self.all_params.extend(module.params)
        for module in self.im_modules:
            self.all_params.extend(module.params)
        # make dist_scale parameter (add it to the inf net parameters)
        self.dist_scale = sharedX( floatX([0.1]) )
        self.all_params.append(self.dist_scale)
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
        # keep a transform that we'll apply to generator output
        if output_transform == 'ident':
            self.output_transform = lambda x: x
        else:
            self.output_transform = output_transform
        # construct a vertically-repeated identity matrix for marginalizing
        # over possible values of the categorical latent variable.
        Ic = np.vstack([np.eye(label_dim) for i in range(self.nbatch)])
        self.Ic = theano.shared(value=floatX(Ic))
        print("Compiling sample generator...")
        # test inputs to sample generator
        y_ind = floatX( np.eye(self.nyc) )
        z0_dim = self.q_z0Iyx_model.modules[-1].rand_chans
        z0_samps = floatX( npr.normal(size=(y_ind.shape[0], z0_dim)) )
        # compile and test sample generating function
        self.generate_samples = self._construct_generate_samples()
        samps = self.generate_samples(z0_samps, y_ind)
        print("DONE.")
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for the top-most inference models
        mod_param_dicts = self.q_aIx_model.dump_params()
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = self.q_yIax_model.dump_params()
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = self.q_z0Iyx_model.dump_params()
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
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
        # reload parameter dicts for the top-most inference models
        mod_param_dicts = cPickle.load(pickle_file)
        self.q_aIx_model.load_params(mod_param_dicts=mod_param_dicts)
        mod_param_dicts = cPickle.load(pickle_file)
        self.q_yIax_model.load_params(mod_param_dicts=mod_param_dicts)
        mod_param_dicts = cPickle.load(pickle_file)
        self.q_z0Iyx_model.load_params(mod_param_dicts=mod_param_dicts)
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

    def apply_td(self, z0, y_ind):
        """
        Apply this generator network using the given random values.

        Assume top-most indicators and rand
        """
        # convolution-appropriate version of indicators
        y_ind_conv = y_ind.dimshuffle(0,1,'x','x')
        td_acts = []
        for td_mod_num, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            if (td_mod_num == 0):
                # this is the top-most, fully-connected module. its inputs
                # are just z0 and y_ind (concatenated horizontally)
                rvs_and_inds = T.concatenate([z0, y_ind], axis=1)
                td_act_i = td_module.apply(rand_vals=rvs_and_inds)
            elif td_mod_name in self.merge_info:
                # handle computation for an internal TD module that requires
                # sampling some latent variables, and TD/BU info merging.
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                im_module = self.im_modules_dict[im_mod_name]
                # sample values to reparametrize, if none given
                batch_size = td_acts[-1].shape[0]
                row_dim = td_acts[-1].shape[2]
                col_dim = td_acts[-1].shape[3]
                rand_chans = im_module.rand_chans
                rvs_size = (batch_size, rand_chans, row_dim, col_dim)
                rvs = cu_rng.normal(size=rvs_size, dtype=theano.config.floatX)
                # concatenate top-down activations with indicators
                td_info = conv_cond_concat(td_acts[-1], y_ind_conv)
                # feedforward through td_module
                td_act_i = td_module.apply(td_info,
                                           rand_vals=rvs)
            else:
                # handle computation for a TD module that only requires
                # info from preceding TD modules (i.e. no rands or indicators)
                td_act_i = td_module.apply(input=td_acts[-1], rand_vals=None)
            td_acts.append(td_act_i)
        # apply some transform (e.g. tanh or sigmoid) to final activations
        result = self.output_transform(td_acts[-1])
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

    def apply_im_y_marginalized_1(self, input):
        """
        This version repeats the input after sampling q(a|x) and q(y|a,x).
        """
        y_ind = self.Ic
        y_ind_conv = self.Ic.dimshuffle(0,1,'x','x')
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input)
        x_info = T.flatten(bu_res_dict['bu_acts'][-1], 2)
        # draw a sample from q(a | x) for each input
        a_cond_mean, a_cond_logvar = self.q_aIx_model.apply(x_info)
        a_cond_mean = self.dist_scale[0] * tanh_clip(a_cond_mean, bound=3.0)
        a_cond_logvar = self.dist_scale[0] * tanh_clip(a_cond_logvar, bound=3.0)
        a_samps = reparametrize(a_cond_mean, a_cond_logvar, rng=cu_rng)
        kld_a = T.sum(gaussian_kld(T.flatten(a_cond_mean, 2),
                                   T.flatten(a_cond_logvar, 2),
                                   0.0, 0.0), axis=1)
        # feed BU features and a samples into q(y | a, x)
        ax_info = T.concatenate([x_info, a_samps], axis=1)
        y_unnorm, _ = self.q_yIax_model.apply(ax_info)
        y_probs = T.nnet.softmax(y_unnorm)
        ent_y = -1.0 * T.sum((y_probs * T.log(y_probs)), axis=1)
        kld_y = self.log_nyc - ent_y
        # repeat the input for marginalizing remaining inference steps
        x_info_rpt = T.extra_ops.repeat(x_info, self.nyc, axis=0)
        # sample from q(z | y, x) for the repeated inputs
        yx_info = T.concatenate([y_ind, x_info_rpt], axis=1)
        z0_cond_mean, z0_cond_logvar = self.q_zIyx_model.apply(yx_info)
        z0_samps = reparametrize(z0_cond_mean, z0_cond_logvar, rng=cu_rng)
        kld_z0 = gaussian_kld(T.flatten(z0_cond_mean, 2),
                              T.flatten(z0_cond_logvar, 2),
                              0.0, 0.0)
        # now, run the TD/BU info merging process through the convolutional
        # modules in this network
        td_acts = []
        td_klds = [ T.sum(kld_z0, axis=1) ]
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            if (i == 0):
                # run through the top-most, fully-connected module
                z0_and_inds = T.concatenate([z0_samps, y_ind], axis=1)
                td_acts_i = td_module.apply(rand_vals=z0_and_inds)
                td_acts.append(td_acts_i)
            elif (td_mod_name in self.merge_info):
                # handle computation for a TD module that requires samples from
                # a conditional distribution formed by merging BU and TD info.
                bu_mod_name = self.merge_info[td_mod_name]['bu_module']
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                im_module = self.im_modules_dict[im_mod_name]
                # handle conditionals based on merging BU and TD info
                td_info = td_acts[-1]              # info from TD pass
                bu_info = bu_res_dict[bu_mod_name] # info from BU pass
                # repeat and/or concatenate info as required
                td_info_and_inds = conv_cond_concat(td_info, y_ind_conv)
                bu_info_rpt = T.extra_ops.repeat(bu_info, self.nyc, axis=0)
                # get the inference distribution using TD/BU info and indicators.
                # the BU info has to be repeated to allow marginalization.
                zi_cond_mean, zi_cond_logvar = \
                        im_module.apply_im(td_input=td_info_and_inds,
                                           bu_input=bu_info_rpt)
                zi_cond_mean = self.dist_scale[0] * tanh_clip(zi_cond_mean, bound=3.0)
                zi_cond_logvar = self.dist_scale[0] * tanh_clip(zi_cond_logvar, bound=3.0)
                # reparametrize and sample from conditional over zi
                zi_samps = reparametrize(zi_cond_mean, zi_cond_logvar,
                                         rng=cu_rng)
                # record KLd for current conditional distribution over zi
                kld_zi = gaussian_kld(T.flatten(zi_cond_mean, 2),
                                      T.flatten(zi_cond_logvar, 2),
                                      0.0, 0.0)
                td_klds.append(T.sum(kld_zi, axis=1))
                # feedforward through the current TD module
                td_acts_i = td_module.apply(input=td_info_and_inds,
                                            rand_vals=zi_samps)
                td_acts.append(td_acts_i)
            else:
                # handle computation for a TD module that only requires info
                # from preceding TD modules (i.e. no rands or indicators)
                td_info = td_acts[-1]
                td_acts_i = td_module.apply(input=td_info, rand_vals=None)
                td_acts.append(td_acts_i)
        # compute nll costs for these outputs
        x_rpt = T.extra_ops.repeat(input, self.nyc, axis=0)
        x_recon_rpt = self.output_transform(td_acts[-1])
        if self.use_bernoulli:
            # use bernoulli output cost
            log_p_xIz_rpt = T.sum(log_prob_bernoulli(T.flatten(x_rpt,2),
                                    T.flatten(x_recon_rpt,2),
                                    do_sum=False), axis=1)
        else:
            # use gaussian(ish) output cost
            log_p_xIz_rpt = T.sum(log_prob_gaussian(T.flatten(x_rpt,2),
                                    T.flatten(x_recon_rpt,2),
                                    log_vars=0.0, use_huber=0.5,
                                    do_sum=False), axis=1)
        # compute total KLd for the merged TD/BU inference
        kld_z_rpt = sum(td_klds)
        # reshape repeated costs, and marginalize w.r.t. y_probs
        log_p_xIz_mat = log_p_xIz_rpt.reshape((self.nbatch, self.nyc))
        kld_z_mat = kld_z_rpt.reshape((self.nbatch, self.nyc))
        log_p_xIz = T.sum((y_probs * log_p_xIz_mat), axis=1)
        kld_z = T.sum((y_probs * kld_z_mat), axis=1)

        # compute overall per-observation costs
        obs_nlls = log_p_xIz
        obs_klds = kld_z + kld_a + kld_y

        # package results for convenient processing
        im_res_dict = {}
        im_res_dict['obs_nlls'] = obs_nlls
        im_res_dict['obs_klds'] = obs_klds
        im_res_dict['log_p_xIz'] = log_p_xIz
        im_res_dict['kld_z'] = kld_z
        im_res_dict['kld_a'] = kld_a
        im_res_dict['kld_y'] = kld_y
        im_res_dict['ent_y'] = ent_y
        return im_res_dict

    def _construct_generate_samples(self):
        """
        Generate some samples from this network.
        """
        z0 = T.matrix()
        y_ind = T.matrix()
        # feedforward through the model, generating conditioned on the given
        # matrix of one-hot indicator vectors
        sym_samples = self.apply_td(z0=z0, y_ind=y_ind)
        # compile a theano function for sampling outputs from the top-down
        # generative process.
        sample_func = theano.function([z0, y_ind], sym_samples)
        return sample_func


















##############
# EYE BUFFER #
##############
