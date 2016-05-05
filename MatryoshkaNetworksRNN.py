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


class DeepSeqCondGen(object):
    '''
    A deep, hierarchical, conditional generator network. This provides a
    wrapper around bottom-up, top-down, and info-merging recurrent modules.

    Params:
        td_modules: modules for computing top-down information.
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
                    -- gen and inf modules should have matching names.
    '''
    def __init__(self,
                 td_modules,
                 bu_modules_gen, im_modules_gen,
                 bu_modules_inf, im_modules_inf,
                 merge_info):
        # grab the bottom-up, top-down, and info merging modules
        self.td_modules = [m for m in td_modules]
        self.bu_modules_gen = [m for m in bu_modules_gen]
        self.im_modules_gen = [m for m in im_modules_gen]
        self.bu_modules_inf = [m for m in bu_modules_inf]
        self.im_modules_inf = [m for m in im_modules_inf]
        # get dicts for referencing modules by name
        self.td_modules_dict = {m.mod_name: m for m in td_modules}
        self.td_modules_dict[None] = None
        self.bu_modules_gen_dict = {m.mod_name: m for m in bu_modules_gen}
        self.bu_modules_gen_dict[None] = None
        self.bu_modules_inf_dict = {m.mod_name: m for m in bu_modules_inf}
        self.bu_modules_inf_dict[None] = None
        self.im_modules_gen_dict = {m.mod_name: m for m in im_modules_gen}
        self.im_modules_gen_dict[None] = None
        self.im_modules_inf_dict = {m.mod_name: m for m in im_modules_inf}
        self.im_modules_inf_dict[None] = None
        # grab the full set of trainable parameters in these modules
        self.gen_params = []  # modules that aren't just for inference
        self.inf_params = []  # modules that are just for inference
        # get generator params (these only get to adapt to the training set)
        self.generator_modules = self.td_modules + self.bu_modules_gen + \
            self.im_modules_gen
        for mod in self.generator_modules:
            self.gen_params.extend(mod.params)
        # get inferencer params (these can be fine-tuned at test time)
        self.inferencer_modules = self.bu_modules_inf + self.im_modules_inf
        for mod in self.inferencer_modules:
            self.inf_params.extend(mod.params)
        # filter redundant parameters, to allow parameter sharing
        p_dict = {}
        for p in self.gen_params:
            p_dict[p.name] = p
        self.gen_params = p_dict.values()
        p_dict = {}
        for p in self.inf_params:
            p_dict[p.name] = p
        self.inf_params = p_dict.values()
        # add a distribution scaling parameter to the generator
        self.dist_scale = sharedX(floatX([0.2]))
        self.gen_params.append(self.dist_scale)
        # gather a list of all parameters in this network
        self.all_params = self.inf_params + self.gen_params
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
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
        '''
        Dump params to a file for later reloading by self.load_params.
        '''
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the parameter dicts for generator modules
        mod_param_dicts = [m.dump_params() for m in self.generator_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        # dump the parameter dicts for inferencer modules
        mod_param_dicts = [m.dump_params() for m in self.inferencer_modules]
        cPickle.dump(mod_param_dicts, f_handle, protocol=-1)
        # dump dist_scale parameter
        ds_ary = self.dist_scale.get_value(borrow=False)
        cPickle.dump(ds_ary, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_params(self, f_name=None):
        '''
        Load params from a file saved via self.dump_params.
        '''
        assert(not (f_name is None))
        pickle_file = open(f_name)
        # reload the parameter dicts for generator modules
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.generator_modules):
            mod.load_params(param_dict=param_dict)
        # reload the parameter dicts for inferencer modules
        mod_param_dicts = cPickle.load(pickle_file)
        for param_dict, mod in zip(mod_param_dicts, self.inferencer_modules):
            mod.load_params(param_dict=param_dict)
        # load dist_scale parameter
        ds_ary = cPickle.load(pickle_file)
        self.dist_scale.set_value(floatX(ds_ary))
        pickle_file.close()
        return

    def apply_mlp(self, input, modules):
        '''
        Apply a sequence of modules to an input -- a quick and dirty MLP.
        '''
        mlp_acts = []
        res_dict = {}
        for i, mod in enumerate(modules):
            if (i == 0):
                mlp_info = input
            else:
                mlp_info = mlp_acts[i - 1]
            res = mod.apply(mlp_info)
            mlp_acts.append(res)
            res_dict[mod.mod_name] = res
        # res_dict returns MLP layer outputs keyed by module name, and in a
        # simple ordered list...
        res_dict['acts'] = mlp_acts
        return res_dict

    def apply_im(self,
                 input_gen,
                 input_inf,
                 td_states=None,
                 im_states_gen=None,
                 im_states_inf=None):
        '''
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        -- this does all the heavy lifting --

        Inputs:
            input_gen: input to the generator upsy-downsy network
            input_inf: input to the inferencer upsy-downsy network
            td_states: shared TD module states at time t - 1
            im_states_gen: generator IM module states at time t - 1
            im_states_inf: inferencer IM module states at time t - 1
        '''
        # gather initial states for stateful modules if none were given
        batch_size = input_gen.shape[0]
        if td_states is None:
            td_states = {tdm.mod_name: tdm.get_s0_for_batch(batch_size)
                         for tdm in self.td_modules}
        if im_states_gen is None:
            im_states_gen = {imm.mod_name: imm.get_s0_for_batch(batch_size)
                             for imm in self.im_modules_gen}
        if im_states_inf is None:
            im_states_inf = {imm.mod_name: imm.get_s0_for_batch(batch_size)
                             for imm in self.im_modules_inf}
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        z_dict = {}        # latent samples used in each TD module
        kld_dict = {}      # KL(inf || gen) in each TD module
        log_pz_dict = {}   # log p(z | x) for each TD module
        log_qz_dict = {}   # log q(z | x) for each TD module
        # first, run the bottom-up passes for generator and inferencer
        bu_res_dict_gen = self.apply_mlp(input=input_gen,
                                         modules=self.bu_modules_gen)
        bu_res_dict_inf = self.apply_mlp(input=input_inf,
                                         modules=self.bu_modules_inf)
        # dicts for storing updated module states
        td_states_new = {}
        im_states_gen_new = {}
        im_states_inf_new = {}
        # now, run top-down pass using latent variables sampled from
        # conditional distributions constructed by merging bottom-up and
        # top-down information.
        td_outs = []
        for i, td_module in enumerate(self.td_modules):
            # get info about this TD module's connections
            td_mod_name = td_module.mod_name
            td_mod_type = self.merge_info[td_mod_name]['td_type']
            im_mod_name = self.merge_info[td_mod_name]['im_module']
            bu_mod_name = self.merge_info[td_mod_name]['bu_module']
            # get states and inputs for processing this TD module
            assert (td_mod_type in ['top', 'cond'])
            if td_mod_type == 'top':
                # use a "dummy" TD input at the top TD module
                td_input = 0. * td_states[td_mod_name]
            else:
                # use previous TD module output at other TD modules
                td_input = td_outs[-1]  # from step t
            td_state = td_states[td_mod_name]            # from step t - 1
            bu_input_gen = bu_res_dict_gen[bu_mod_name]  # from step t
            bu_input_inf = bu_res_dict_inf[bu_mod_name]  # from step t
            im_state_gen = im_states_gen[im_mod_name]    # from step t - 1
            im_state_inf = im_states_inf[im_mod_name]    # from step t - 1
            # get IM modules to apply at this step
            im_module_gen = self.im_modules_gen_dict[im_mod_name]
            im_module_inf = self.im_modules_inf_dict[im_mod_name]
            # get conditional Gaussian parameters from generator
            cond_mean_gen, cond_logvar_gen, im_state_gen_new = \
                im_module_gen.apply_im(state=im_state_gen,
                                       td_state=td_state,
                                       td_input=td_input,
                                       bu_input=bu_input_gen)
            cond_mean_gen = self.dist_scale[0] * cond_mean_gen
            cond_logvar_gen = self.dist_scale[0] * cond_logvar_gen
            # get conditional Gaussian parameters from inferencer
            cond_mean_inf, cond_logvar_inf, im_state_inf_new = \
                im_module_inf.apply_im(state=im_state_inf,
                                       td_state=td_state,
                                       td_input=td_input,
                                       bu_input=bu_input_inf)
            cond_mean_inf = self.dist_scale[0] * cond_mean_inf
            cond_logvar_inf = self.dist_scale[0] * cond_logvar_inf
            # do reparametrization for gen and inf models
            cond_z_gen = reparametrize(cond_mean_gen, cond_logvar_gen,
                                       rng=cu_rng)
            cond_z_inf = reparametrize(cond_mean_inf, cond_logvar_inf,
                                       rng=cu_rng)
            cond_z = (self.sample_switch[0] * cond_z_inf) + \
                ((1. - self.sample_switch[0]) * cond_z_gen)
            # update the current TD module
            td_output, td_state_new = \
                td_module.apply(state=td_state, input=td_input, rand_vals=cond_z)
            # get KL divergence between inferencer and generator
            kld_z = gaussian_kld(T.flatten(cond_mean_inf, 2),
                                 T.flatten(cond_logvar_inf, 2),
                                 T.flatten(cond_mean_gen, 2),
                                 T.flatten(cond_logvar_gen, 2))
            # get the log likelihood of the current latent samples under
            # both the proposal distribution q(z | x) and the prior p(z).
            # -- these are used when computing the IWAE bound.
            log_pz = log_prob_gaussian(T.flatten(cond_z, 2),
                                       T.flatten(cond_mean_gen, 2),
                                       log_vars=T.flatten(cond_logvar_gen, 2),
                                       do_sum=True)
            # get the log likelihood of z under a default prior.
            log_qz = log_prob_gaussian(T.flatten(cond_z, 2),
                                       T.flatten(cond_mean_inf, 2),
                                       log_vars=T.flatten(cond_logvar_inf, 2),
                                       do_sum=True)
            # record values produced while processing this TD module
            # -- these values are all keyed by the TD module name
            td_outs.append(td_output)
            td_states_new[td_mod_name] = td_state_new
            im_states_gen_new[im_mod_name] = im_state_gen_new
            im_states_inf_new[im_mod_name] = im_state_inf_new
            z_dict[td_mod_name] = cond_z
            kld_dict[td_mod_name] = kld_z
            log_pz_dict[td_mod_name] = log_pz
            log_qz_dict[td_mod_name] = log_qz

        # We return:
        #   1. the canvas update generated by this step
        #   2. the updated TD module states
        #   3. the updated IM module states for gen/inf
        #   4. the latent samples used in this canvas update
        #   5. the KL(q || p) for each TD module
        #   6. the log q(z|x) and log p(z|x) for each TD module
        res_dict = {}
        res_dict['output'] = td_outs[-1]
        res_dict['td_states'] = td_states_new
        res_dict['im_states_gen'] = im_states_gen_new
        res_dict['im_states_inf'] = im_states_inf_new
        res_dict['z_dict'] = z_dict
        res_dict['kld_dict'] = kld_dict
        res_dict['log_pz_dict'] = log_pz_dict
        res_dict['log_qz_dict'] = log_qz_dict
        return res_dict





##############
# EYE BUFFER #
##############
