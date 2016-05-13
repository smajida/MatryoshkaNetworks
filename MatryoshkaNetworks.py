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
        self.bu_modules_dict = {m.mod_name: m for m in bu_modules}
        self.bu_modules_dict[None] = None
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
        result = None
        dec_count = 0.
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
                    if hasattr(td_module, 'has_decoder'):
                        # TD modules with auxiliary decoder return two tensors
                        td_act_i, dec_res_i = td_act_i
                        dec_count = dec_count + 1.
                        if result is None:
                            # first decoder initializes output
                            result = dec_res_i
                        else:
                            # final output is average of the decoders' outputs
                            scale = 1. / dec_count
                            result = (1. - scale) * result + (scale * dec_res_i)
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules (no rand input)
                td_act_i = td_module.apply(input=td_acts[-1], rand_vals=None,
                                           noise=td_noise)
            td_acts.append(td_act_i)
        # apply some transform (e.g. tanh or sigmoid) to final activations
        if result is None:
            result = self.output_transform(td_acts[-1])
        else:
            result = self.output_transform(result)
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
        td_output = None
        dec_count = 0.
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            bu_src_name = self.merge_info[td_mod_name]['bu_source']
            im_src_name = self.merge_info[td_mod_name]['im_source']
            im_module = self.im_modules_dict[im_mod_name]  # this might be None
            unif_post = False
            if td_mod_type in ['top', 'cond']:
                if td_mod_type == 'top':
                    # top TD conditionals are based purely on BU info
                    bu_module = self.bu_modules_dict[bu_src_name]
                    cond_mean_im = bu_res_dict[bu_src_name][0]
                    cond_logvar_im = bu_res_dict[bu_src_name][1]
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    cond_mean_td = 0.0 * cond_mean_im
                    cond_logvar_td = 0.0 * cond_logvar_im
                    if bu_module.unif_post is None:
                        # reparametrize Gaussian for latent samples
                        cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                                 rng=cu_rng)
                    else:
                        # use uniform reparametrization with bounded KLd
                        cond_rvs, kld_i, log_p_z, log_q_z = \
                            reparametrize_uniform(cond_mean_im, cond_logvar_im,
                                                  scale=bu_module.unif_post,
                                                  rng=cu_rng)
                        unif_post = True
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

                    if im_module.unif_post is None:
                        # reparametrize Gaussian for latent samples
                        cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                                 rng=cu_rng)
                    else:
                        # use uniform reparametrization with bounded KLd
                        cond_rvs, kld_i, log_p_z, log_q_z = \
                            reparametrize_uniform(cond_mean_im, cond_logvar_im,
                                                  scale=im_module.unif_post,
                                                  rng=cu_rng)
                        unif_post = True
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=cond_rvs,
                                               noise=td_noise)
                    if hasattr(td_module, 'has_decoder'):
                        # TD modules with auxiliary decoder return two tensors
                        td_act_i, dec_res_i = td_act_i
                        dec_count = dec_count + 1.
                        if td_output is None:
                            # first decoder initializes output
                            td_output = dec_res_i
                        else:
                            # final output is average of the decoders' outputs
                            # scale = 1. / dec_count
                            # td_output = (1. - scale) * td_output + (scale * dec_res_i)
                            td_output = td_output + dec_res_i
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict[im_mod_name] = im_act_i
                if not unif_post:
                    # record KLd info for the conditional distribution
                    kld_i = gaussian_kld(T.flatten(cond_mean_im, 2),
                                         T.flatten(cond_logvar_im, 2),
                                         T.flatten(cond_mean_td, 2),
                                         T.flatten(cond_logvar_td, 2))
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
                kld_dict[td_mod_name] = kld_i
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
        if td_output is None:
            td_output = self.output_transform(td_acts[-1])
        else:
            td_output = self.output_transform(td_output)
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
        for module in self.td_modules:
            self.gen_params.extend(module.params)
        for module in self.bu_modules_gen:
            self.gen_params.extend(module.params)
        for module in self.im_modules_gen:
            self.gen_params.extend(module.params)
        for module in self.bu_modules_inf:
            self.inf_params.extend(module.params)
        for module in self.im_modules_inf:
            self.inf_params.extend(module.params)
        # filter redundant parameters, to allow parameter sharing
        p_dict = {}
        for p in self.gen_params:
            p_dict[p.auto_name] = p
        self.gen_params = p_dict.values()
        p_dict = {}
        for p in self.inf_params:
            p_dict[p.auto_name] = p
        self.inf_params = p_dict.values()
        p_dict = {}
        for p in self.gen_params:
            p_dict[p.auto_name] = p
        for p in self.inf_params:
            p_dict[p.auto_name] = p
        self.all_params = p_dict.values()
        # make dist_scale parameter (add it to the inf net parameters)
        if train_dist_scale:
            # init to a somewhat arbitrary value -- not magic (probably)
            self.dist_scale = sharedX(floatX([0.2]))
            self.gen_params.append(self.dist_scale)
        else:
            self.dist_scale = sharedX(floatX([1.0]))
        # gather a list of all parameters in this network
        self.params = self.all_params
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
        # keep a transform that we'll apply to generator output
        if output_transform == 'ident':
            self.output_transform = lambda x: x
        else:
            self.output_transform = output_transform
        # make a switch for alternating between generator and inferencer
        # conditionals over the latent variables
        self.sample_switch = sharedX(floatX([1.0]))
        return

    def set_sample_switch(self, source='inf'):
        '''
        Set the latent sample switch to use samples from the given source.
        '''
        assert (source_name in ['inf', 'gen'])
        switch_val = floatX([1.]) if (source_name == 'inf') else floatX([0.])
        self.sample_switch.set_value(switch_val)
        return

    def dump_params(self, f_name=None):
        """
        Dump params to a file for later reloading by self.load_params.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for all modules in this network
        mod_param_dicts = [m.dump_params() for m in self.td_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = [m.dump_params() for m in self.bu_modules_gen]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = [m.dump_params() for m in self.im_modules_gen]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = [m.dump_params() for m in self.bu_modules_inf]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        mod_param_dicts = [m.dump_params() for m in self.im_modules_inf]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
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
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.td_modules):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.bu_modules_gen):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.im_modules_gen):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.bu_modules_inf):
            mod.load_params(param_dict=param_dict)
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.im_modules_inf):
            mod.load_params(param_dict=param_dict)
        # load dist_scale parameter
        ds_ary = cPickle.load(pickle_file)
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def apply_bu(self, input, mode='gen'):
        '''
        Apply this model's bottom-up inference modules to the given input,
        and return a dict mapping BU module names to their outputs.
        -- mode can be either 'gen' or 'inf'.
        -- mode determines which BU modules will be used.
        '''
        if mode == 'gen':
            bu_modules = self.bu_modules_gen
        else:
            bu_modules = self.bu_modules_inf
        bu_acts = []
        res_dict = {}
        for i, bu_mod in enumerate(bu_modules):
            if (i == 0):
                bu_info = input
            else:
                bu_info = bu_acts[i - 1]
            res = bu_mod.apply(bu_info)
            bu_acts.append(res)
            res_dict[bu_mod.mod_name] = res
        res_dict['bu_acts'] = bu_acts
        return res_dict

    def apply_im(self, input_gen, input_inf):
        '''
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        ++ this does all the heavy lifting ++

        This first computes the full bottom-up pass to collect the output of
        each BU module, where the output of the final BU module is the means
        and log variances for a diagonal Gaussian distribution over the latent
        variables that will be fed as input to the first TD module.

        This then computes the top-down pass using latent variables sampled
        from distributions determined by merging partial results of the BU pass
        with results from the partially-completed TD pass. The IM modules can
        feed information to eachother too.
        '''
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        z_dict = {}
        kld_dict = {}
        log_pz_dict = {}
        log_qz_dict = {}
        # first, run the bottom-up pass
        bu_res_dict_gen = self.apply_bu(input=input_gen, mode='gen')
        bu_res_dict_inf = self.apply_bu(input=input_inf, mode='inf')
        # dict for storing IM state information
        im_res_dict_gen = {None: None}
        im_res_dict_inf = {None: None}
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
            im_module_gen = self.im_modules_gen_dict[im_mod_name]
            im_module_inf = self.im_modules_inf_dict[im_mod_name]
            if td_mod_type in ['top', 'cond']:
                if td_mod_type == 'top':
                    # get conditional from generator
                    cond_mean_gen = bu_res_dict_gen[bu_src_name][0]
                    cond_logvar_gen = bu_res_dict_gen[bu_src_name][1]
                    cond_mean_gen = self.dist_scale[0] * cond_mean_gen
                    cond_logvar_gen = self.dist_scale[0] * cond_logvar_gen
                    # get conditional from inferencer
                    cond_mean_inf = bu_res_dict_inf[bu_src_name][0]
                    cond_logvar_inf = bu_res_dict_inf[bu_src_name][1]
                    cond_mean_inf = self.dist_scale[0] * cond_mean_inf
                    cond_logvar_inf = self.dist_scale[0] * cond_logvar_inf
                    # generate new latent samples via reparametrization
                    z_gen = reparametrize(cond_mean_gen, cond_logvar_gen,
                                          rng=cu_rng)
                    z_inf = reparametrize(cond_mean_inf, cond_logvar_inf,
                                          rng=cu_rng)
                    z_vals = (self.sample_switch[0] * z_inf) + \
                        ((1. - self.sample_switch[0]) * z_gen)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(rand_vals=z_vals)
                    # compute initial state for IM pass, maybe...
                    im_act_gen = None
                    im_act_inf = None
                    if (im_module_gen is not None):
                        im_act_gen = im_module_gen.apply(rand_vals=z_vals)
                        im_act_inf = im_module_inf.apply(rand_vals=z_vals)
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]
                    bu_info_gen = bu_res_dict_gen[bu_src_name]
                    im_info_gen = im_res_dict_gen[im_src_name]
                    bu_info_inf = bu_res_dict_inf[bu_src_name]
                    im_info_inf = im_res_dict_inf[im_src_name]
                    # get conditional from the generator
                    cond_mean_gen, cond_logvar_gen, im_act_gen = \
                        im_module_gen.apply_im(td_input=td_info,
                                               bu_input=bu_info_gen,
                                               im_input=im_info_gen)
                    cond_mean_gen = self.dist_scale[0] * cond_mean_gen
                    cond_logvar_gen = self.dist_scale[0] * cond_logvar_gen
                    # get conditional from the inferencer
                    cond_mean_inf, cond_logvar_inf, im_act_inf = \
                        im_module_inf.apply_im(td_input=td_info,
                                               bu_input=bu_info_inf,
                                               im_input=im_info_inf)
                    cond_mean_inf = self.dist_scale[0] * cond_mean_inf
                    cond_logvar_inf = self.dist_scale[0] * cond_logvar_inf
                    # generate new latent samples via reparametrization
                    z_gen = reparametrize(cond_mean_gen, cond_logvar_gen,
                                          rng=cu_rng)
                    z_inf = reparametrize(cond_mean_inf, cond_logvar_inf,
                                          rng=cu_rng)
                    z_vals = (self.sample_switch[0] * z_inf) + \
                        ((1. - self.sample_switch[0]) * z_gen)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=z_vals)
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict_gen[im_mod_name] = im_act_gen
                im_res_dict_inf[im_mod_name] = im_act_inf
                # get KL divergence between inferencer and generator
                kld_z = gaussian_kld(T.flatten(cond_mean_inf, 2),
                                     T.flatten(cond_logvar_inf, 2),
                                     T.flatten(cond_mean_gen, 2),
                                     T.flatten(cond_logvar_gen, 2))
                kld_z = T.sum(kld_z, axis=1)
                # get the log likelihood of the current latent samples under
                # both the proposal distribution q(z | x) and the prior p(z).
                # -- these are used when computing the IWAE bound.
                log_pz = log_prob_gaussian(T.flatten(z_vals, 2),
                                           T.flatten(cond_mean_gen, 2),
                                           log_vars=T.flatten(cond_logvar_gen, 2),
                                           do_sum=True)
                # get the log likelihood of z under a default prior.
                log_qz = log_prob_gaussian(T.flatten(z_vals, 2),
                                           T.flatten(cond_mean_inf, 2),
                                           log_vars=T.flatten(cond_logvar_inf, 2),
                                           do_sum=True)
                # record latent samples and loglikelihood for current TD module
                z_dict[td_mod_name] = z_vals
                kld_dict[td_mod_name] = kld_z
                log_pz_dict[td_mod_name] = log_pz
                log_qz_dict[td_mod_name] = log_qz
            elif td_mod_type == 'pass':
                # handle computation for a TD module that only requires
                # information from preceding TD modules
                td_info = td_acts[-1]
                td_act_i = td_module.apply(input=td_info, rand_vals=None)
                td_acts.append(td_act_i)
                if not (im_module_gen is None):
                    # perform an update of the IM state (for gen and inf)
                    im_info_gen = im_res_dict_gen[im_src_name]
                    im_res_dict_gen[im_mod_name] = \
                        im_module_gen.apply(input=im_info_gen, rand_vals=None)
                    im_info_inf = im_res_dict_inf[im_src_name]
                    im_res_dict_inf[im_mod_name] = \
                        im_module_inf.apply(input=im_info_inf, rand_vals=None)
            else:
                assert False, "BAD td_mod_type: {}".format(td_mod_type)
        # apply output transform (into observation space, presumably), to get
        # the final "reconstruction" produced by the merged BU/TD pass.
        output = self.output_transform(td_acts[-1])
        # package results into a handy dictionary
        im_res_dict = {}
        im_res_dict['output'] = output
        im_res_dict['td_acts'] = td_acts
        im_res_dict['z_dict'] = z_dict
        im_res_dict['kld_dict'] = kld_dict
        im_res_dict['log_pz_dict'] = log_pz_dict
        im_res_dict['log_qz_dict'] = log_qz_dict
        return im_res_dict

    # def _construct_generate_samples(self):
    #     """
    #     Generate some samples from this network.
    #     """
    #     batch_size = T.lscalar()
    #     # feedforward through the model with batch size "batch_size"
    #     sym_samples = self.apply_gen(batch_size=batch_size)
    #     # compile a theano function for sampling outputs from the top-down
    #     # generative model.
    #     sample_func = theano.function([batch_size], sym_samples)
    #     return sample_func


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


class InfGenModelGMM(object):
    """
    A deep, hierarchical generator network. This provides a wrapper around a
    collection of bottom-up, top-down, and info-merging Matryoshka modules.

    Params:
        bu_modules: modules for computing bottom-up (inference) information.
        td_modules: modules for computing top-down (generative) information.
        im_modules: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        mix_module: that represents a mixture prior over the top-most latents
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
        output_transform: transform to apply to outputs of the top-down model.
        use_td_noise: whether to apply noise in TD modules
        use_bu_noise: whether to apply noise in BU/IM modules
        train_dist_scale: whether to train rescaling param (for testing)
    """
    def __init__(self,
                 bu_modules, td_modules, im_modules, mix_module,
                 merge_info, output_transform,
                 use_td_noise=False,
                 use_bu_noise=False,
                 train_dist_scale=True):
        # grab the bottom-up, top-down, and info merging modules
        self.bu_modules = [m for m in bu_modules]
        self.td_modules = [m for m in td_modules]
        self.im_modules = [m for m in im_modules]
        self.mix_module = mix_module
        self.im_modules_dict = {m.mod_name: m for m in im_modules}
        self.im_modules_dict[None] = None
        # grab the full set of trainable parameters in these modules
        self.gen_params = []
        self.inf_params = []
        for module in self.td_modules:
            self.gen_params.extend(module.params)
        for module in self.bu_modules:
            self.inf_params.extend(module.params)
        for module in self.im_modules:
            self.inf_params.extend(module.params)
        self.gen_params.extend(mix_module.params)
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
        # dump class module params
        cPickle.dump(self.mix_module.dump_params(), f_handle, protocol=-1)
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
        # load class module params
        self.mix_module.load_params(param_dict=cPickle.load(pickle_file))
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

    def apply_bu(self, input, noise=None):
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
            res = bu_mod.apply(bu_info, noise=bu_noise)
            bu_acts.append(res)
            res_dict[bu_mod.mod_name] = res
        res_dict['bu_acts'] = bu_acts
        return res_dict

    def apply_im(self, input, kl_mode='analytical', noise=None):
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
        mix_post_ent = None
        mix_comp_weight = None
        logz_dict = {'log_p_z': [], 'log_q_z': []}
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input=input, noise=bu_noise)
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
                    # -- in SS model, this also gives class predictions
                    td_act_i = td_module.apply(rand_vals=cond_rvs,
                                               noise=td_noise)
                    # compute initial state for IM pass, maybe...
                    im_act_i = None
                    if not (im_module is None):
                        im_act_i = im_module.apply(rand_vals=cond_rvs,
                                                   noise=bu_noise)
                    ###########################################################
                    # Compute log p(z) and KL(q || p) for a GMM prior         #
                    # -- This must return an analytical approximation to the  #
                    #    desired KL, and the exact log probabilities required #
                    #    for an unbiased Monte-Carlo approximation.           #
                    # #########################################################
                    kld_i, log_p_z, log_q_z, mix_post_ent, mix_comp_weight = \
                        self.mix_module.compute_kld_info(cond_mean_im,
                                                         cond_logvar_im,
                                                         cond_rvs)
                    if kl_mode == 'monte-carlo':
                        # use unbiased monte-carlo approximation of KL(q || p)
                        kld_i = log_q_z - log_p_z
                    # spread out the kld, to make it compatible with elem-wise klds
                    kld_i = T.repeat(kld_i.dimshuffle(0, 'x'), cond_rvs.shape[1], axis=1)
                    kld_i = (1. / T.cast(cond_rvs.shape[1], 'floatX')) * kld_i
                else:
                    # handle conditionals based on merging BU and TD info
                    td_info = td_acts[-1]               # info from TD pass
                    bu_info = bu_res_dict[bu_src_name]  # info from BU pass
                    im_info = im_res_dict[im_src_name]  # info from IM pass
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
                    # reparametrize Gaussian for latent samples
                    cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                             rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=cond_rvs,
                                               noise=td_noise)
                    # record KLd info for the conditional distribution
                    kld_i = gaussian_kld(T.flatten(cond_mean_im, 2),
                                         T.flatten(cond_logvar_im, 2),
                                         T.flatten(cond_mean_td, 2),
                                         T.flatten(cond_logvar_td, 2))
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
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict[im_mod_name] = im_act_i
                kld_dict[td_mod_name] = kld_i
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
        im_res_dict['mix_post_ent'] = mix_post_ent
        im_res_dict['mix_comp_weight'] = mix_comp_weight
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











##############
# EYE BUFFER #
##############
