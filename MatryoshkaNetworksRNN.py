import cPickle
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

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

from theano.printing import Print

#
# Phil's business
#
from MatryoshkaNetworks import *

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()


class DeepSeqCondGenRNN(object):
    '''
    A deep, hierarchical, conditional generator network. This provides a
    wrapper around bottom-up, top-down, and info-merging recurrent modules.

    Params:
        td_modules: modules for computing top-down information.
        bu_modules: modules for computing bottom-up information.
        im_modules: modules for merging bottom-up and top-down information
                    to put conditionals over Gaussian latent variables that
                    participate in the top-down computation.
        bu_modules_inf: modules for computing bottom-up (inference) information
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
                    -- gen and inf modules should have matching names.
    '''
    def __init__(self,
                 td_modules,
                 bu_modules,
                 im_modules,
                 bu_modules_inf,
                 merge_info):
        # grab the bottom-up, top-down, and info merging modules
        self.td_modules = [m for m in td_modules]
        self.bu_modules = [m for m in bu_modules]
        self.im_modules = [m for m in im_modules]
        self.bu_modules_inf = [m for m in bu_modules_inf]
        # get dicts for referencing modules by name
        self.td_modules_dict = {m.mod_name: m for m in td_modules}
        self.td_modules_dict[None] = None
        self.bu_modules_dict = {m.mod_name: m for m in bu_modules}
        self.bu_modules_dict[None] = None
        self.im_modules_dict = {m.mod_name: m for m in im_modules}
        self.im_modules_dict[None] = None
        self.bu_modules_inf_dict = {m.mod_name: m for m in bu_modules_inf}
        self.bu_modules_inf_dict[None] = None
        # grab the full set of trainable parameters in these modules
        self.gen_params = []  # modules that aren't just for inference
        self.inf_params = []  # modules that are just for inference
        # get generator params (these only get to adapt to the training set)
        self.generator_modules = self.td_modules + self.bu_modules
        for mod in self.generator_modules:
            self.gen_params.extend(mod.params)
        # get inferencer params (these can be fine-tuned at test time)
        self.inferencer_modules = self.im_modules + self.bu_modules_inf
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
        assert (source in ['inf', 'gen'])
        switch_val = floatX([1.]) if (source == 'inf') else floatX([0.])
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

    def apply_rnn(self, input, states, modules):
        '''
        Apply a sequence of recurrent modules to an input -- quick and dirty.
        '''
        ff_acts = []
        output_dict = {}
        state_dict = {}
        for i, mod in enumerate(modules):
            # get feedforward input to this module
            if (i == 0):
                ff_info = input
            else:
                ff_info = ff_acts[i - 1]
            if states is not None:
                # get state input to this module
                state_info = states[mod.mod_name]
                # apply stateful update in this module
                output, state = mod.apply(state=state_info, input=ff_info)
            else:
                # apply stateless update in this module
                output = mod.apply(input=ff_info)
                state = None
            # record results
            ff_acts.append(output)
            output_dict[mod.mod_name] = output
            state_dict[mod.mod_name] = state
        # return results
        res_dict = {'outputs': output_dict, 'states': state_dict}
        return res_dict

    def apply_im_cond(self,
                      input_gen,
                      input_inf,
                      td_states=None,
                      bu_states=None,
                      im_states=None):
        '''
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        -- this does all the heavy lifting --

        Inputs:
            input_gen: input to the generator
            input_inf: input to the inferencer
            td_states: generator TD module states at time t - 1
            bu_states: generator BU module states at time t - 1
            im_states: inferencer IM module states at time t - 1
        '''
        # gather initial states for stateful modules if none were given
        batch_size = input_gen.shape[0]
        if td_states is None:
            td_states = {mod.mod_name: mod.get_s0_for_batch(batch_size)
                         for mod in self.td_modules}
        if bu_states is None:
            bu_states = {mod.mod_name: mod.get_s0_for_batch(batch_size)
                         for mod in self.bu_modules}
        if im_states is None:
            im_states = {mod.mod_name: mod.get_s0_for_batch(batch_size)
                         for mod in self.im_modules}
        # set aside a dict for recording KLd info at each layer that requires
        # samples from a conditional distribution over the latent variables.
        z_dict = {}        # latent samples used in each TD module
        kld_dict = {}      # KL(inf || gen) in each TD module
        log_pz_dict = {}   # log p(z | x) for each TD module
        log_qz_dict = {}   # log q(z | x) for each TD module
        # first, run the bottom-up passes for generator and inferencer
        # -- this collects activations and state information
        bu_res_dict_gen = self.apply_rnn(input=input_gen,
                                         states=bu_states,
                                         modules=self.bu_modules)
        bu_res_dict_inf = self.apply_rnn(input=input_inf,
                                         states=None,
                                         modules=self.bu_modules_inf)
        # get output activations from the BU modules
        bu_outputs_gen = bu_res_dict_gen['outputs']
        bu_outputs_inf = bu_res_dict_inf['outputs']
        # dicts for storing updated module states
        td_states_new = {}
        im_states_new = {}
        bu_states_new = bu_res_dict_gen['states']  # BU state in generator
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
            im_module = self.im_modules_dict[im_mod_name]
            td_state = td_states[td_mod_name]            # from TD step t - 1
            im_state = im_states[im_mod_name]            # from IM step t - 1
            bu_input_gen = bu_outputs_gen[bu_mod_name]   # from BU step t
            bu_input_inf = bu_outputs_inf[bu_mod_name]   # from BU step t

            # concatenate inferencer and generator BU and TD generator input
            td_input = T.concatenate([td_input, bu_input_gen], axis=1)

            # get conditional Gaussian parameters from inference net
            cond_mean_inf, cond_logvar_inf, im_state_new = \
                im_module.apply_im(state=im_state,
                                   td_state=td_state,
                                   td_input=td_input,
                                   bu_input=bu_input_inf)
            cond_mean_inf = self.dist_scale[0] * cond_mean_inf
            cond_logvar_inf = self.dist_scale[0] * cond_logvar_inf
            cond_mean_gen = 0. * cond_mean_inf
            cond_logvar_gen = 0. * cond_logvar_inf

            # do reparametrization for gen and inf models
            cond_z_gen = reparametrize(cond_mean_gen, cond_logvar_gen,
                                       rng=cu_rng)
            cond_z_inf = reparametrize(cond_mean_inf, cond_logvar_inf,
                                       rng=cu_rng)
            cond_z = (self.sample_switch[0] * cond_z_inf) + \
                ((1. - self.sample_switch[0]) * cond_z_gen)

            use_rand = True
            if hasattr(td_module, 'use_rand'):
                use_rand = td_module.use_rand

            if not use_rand:
                cond_z = 0. * cond_z
                cond_mean_inf = 0. * cond_mean_inf
                cond_mean_gen = 0. * cond_mean_gen

            # update the current TD module using the conditional z samples
            td_output, td_state_new = \
                td_module.apply(state=td_state, input=td_input, rand_vals=cond_z)

            # get KL divergence between inferencer and generator
            kld_z = gaussian_kld(T.flatten(cond_mean_inf, 2),
                                 T.flatten(cond_logvar_inf, 2),
                                 T.flatten(cond_mean_gen, 2),
                                 T.flatten(cond_logvar_gen, 2))
            kld_z = T.sum(kld_z, axis=1)
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
            td_outs.append(td_output)
            td_states_new[td_mod_name] = td_state_new
            im_states_new[im_mod_name] = im_state_new
            z_dict[td_mod_name] = cond_z
            kld_dict[td_mod_name] = kld_z
            log_pz_dict[td_mod_name] = log_pz
            log_qz_dict[td_mod_name] = log_qz

        # We return:
        #   1. the canvas update generated by this step
        #   2. the updated TD module states (for gen)
        #   3. the updated BU module states (for gen)
        #   4. the updated IM module states (for inf)
        #   5. the latent samples used in this canvas update
        #   6. the KL(q || p) for each TD module
        #   7. the log q(z|x) and log p(z|x) for each TD module
        res_dict = {}
        res_dict['output'] = td_outs[-1]
        res_dict['td_states'] = td_states_new
        res_dict['bu_states'] = bu_states_new
        res_dict['im_states'] = im_states_new
        res_dict['z_dict'] = z_dict
        res_dict['kld_dict'] = kld_dict
        res_dict['log_pz_dict'] = log_pz_dict
        res_dict['log_qz_dict'] = log_qz_dict
        return res_dict




##############
# EYE BUFFER #
##############
