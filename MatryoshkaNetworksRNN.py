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
from MatryoshkaNetworks import *

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()


class CondDeepConvGRU(object):
    '''
    A deep, hierarchical, convolutional, conditional GRU network.
    This provides a wrapper around a collection of bottom-up, top-down, and
    info-merging Matryoshka modules.

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
        print("Compiling sample generator...")
        # # i'm the operator with my sample generator
        # self.generate_samples = self._construct_generate_samples()
        # samps = self.generate_samples(32)
        print("DONE.")
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

    def apply_im(self, input, mode='gen', z_vals=None):
        '''
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.
        -- mode can be either 'gen' or 'inf'.
        -- mode determines which BU and IM modules will be used.
        -- If z_vals is given, we use those values for the latent samples in
           the corresponding TD module, and don't draw new samples.

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
        assert (mode in {'gen', 'inf'}), "mode must be in {'gen', 'inf'}"
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        z_dict = {}
        logz_dict = {}
        logy_dict = {}
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input=input, mode=mode)
        # dict for storing IM state information
        im_res_dict = {None: None}
        # grab the appropriate sets of BU and IM modules...
        if mode == 'gen':
            bu_modules = self.bu_modules_gen
            im_modules = self.im_modules_gen
            im_modules_dict = self.im_modules_gen_dict
        else:
            bu_modules = self.bu_modules_inf
            im_modules = self.im_modules_inf
            im_modules_dict = self.im_modules_inf_dict
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
            im_module = im_modules_dict[im_mod_name]  # this might be None
            if td_mod_type in ['top', 'cond']:
                if td_mod_type == 'top':
                    # top TD conditionals are based purely on BU info
                    cond_mean_im = bu_res_dict[bu_src_name][0]
                    cond_logvar_im = bu_res_dict[bu_src_name][1]
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    if z_vals is not None:
                        # use previously sampled latent values
                        cond_rvs = z_vals[td_mod_name]
                    else:
                        # generate new latent samples via reparametrization
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
                    # get the conditional distribution SSs (Sufficient Stat s)
                    cond_mean_im, cond_logvar_im, im_act_i = \
                        im_module.apply_im(td_input=td_info,
                                           bu_input=bu_info,
                                           im_input=im_info)
                    cond_mean_im = self.dist_scale[0] * cond_mean_im
                    cond_logvar_im = self.dist_scale[0] * cond_logvar_im
                    if z_vals is not None:
                        # use previously sampled latent values
                        cond_rvs = z_vals[td_mod_name]
                    else:
                        # generate new latent samples via reparametrization
                        cond_rvs = reparametrize(cond_mean_im, cond_logvar_im,
                                                 rng=cu_rng)
                    # feedforward through the current TD module
                    td_act_i = td_module.apply(input=td_info,
                                               rand_vals=cond_rvs)
                # record top-down activations produced by IM and TD modules
                td_acts.append(td_act_i)
                im_res_dict[im_mod_name] = im_act_i
                # get the log likelihood of the current latent samples under
                # both the proposal distribution q(z | x) and the prior p(z).
                # -- these are used when computing the IWAE bound.
                log_prob_z = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                               T.flatten(cond_mean_im, 2),
                                               log_vars=T.flatten(cond_logvar_im, 2),
                                               do_sum=True)
                # get the log likelihood of z under a default prior.
                log_prob_y = log_prob_gaussian(T.flatten(cond_rvs, 2),
                                               (0. * T.flatten(cond_mean_im, 2)),
                                               log_vars=(0. * T.flatten(cond_logvar_im, 2)),
                                               do_sum=True)
                # record latent samples and loglikelihood for current TD module
                logz_dict[td_mod_name] = log_prob_z
                logy_dict[td_mod_name] = log_prob_y
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
        im_res_dict['td_acts'] = td_acts
        im_res_dict['bu_acts'] = bu_res_dict['bu_acts']
        im_res_dict['z_dict'] = z_dict
        im_res_dict['logz_dict'] = logz_dict
        im_res_dict['logy_dict'] = logy_dict
        return im_res_dict
