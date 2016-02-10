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
from lib.ops import batchnorm, deconv, reparametrize, conv_cond_concat, \
                    mean_pool_rows
from lib.theano_utils import floatX, sharedX
from lib.costs import log_prob_bernoulli, log_prob_gaussian, gaussian_kld

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

def tanh_clip_softmax(x, bound=10.0):
    """
    Do soft "tanh" clipping to put data in range -scale....+scale.
    """
    x_clip = bound * tanh((1.0 / bound) * x)
    x_clip_sm = T.nnet.softmax(x_clip)
    return x_clip_sm

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

########################################################################
# Collection of modules for variational inference and generating stuff #
########################################################################

class InfGenModel2(object):
    """
    A deep convolutional generator network. This provides a wrapper around a
    collection of bottom-up, top-down, and info merging Matryoshka modules.

    Params:
        bu_modules: modules for computing bottom-up (inference) information.
        td_modules: modules for computing top-down (generative) information.
        im_modules: modules for merging bottom-up and top-down information
                    sample conditional Gaussian to apply conditional
                    perturbations to the top-down information.
        merge_info: dict of dicts describing how to compute the conditionals
                    required by the feedforward pass through top-down modules.
    """
    def __init__(self,
                 bu_modules, td_modules, im_modules,
                 merge_info):
        # grab the bottom-up, top-down, and info merging modules
        self.bu_modules = [m for m in bu_modules]
        self.td_modules = [m for m in td_modules]
        self.im_modules = [m for m in im_modules]
        self.im_module_dict = {m.mod_name: m for m in im_modules}
        self.bu_module_dict = {m.mod_name: m for m in bu_modules}
        self.td_module_dict = {m.mod_name: m for m in td_modules}
        # grab the full set of trainable parameters in these modules
        self.all_params = []
        for module in self.td_modules: # top-down is the generator
            self.all_params.extend(module.params)
        for module in self.bu_modules: # bottom-up is part of inference
            self.all_params.extend(module.params)
        for module in self.im_modules: # info merge is part of inference
            self.all_params.extend(module.params)
        # make dist_scale parameter (add it to the inf net parameters)
        self.dist_scale = sharedX( floatX([0.1]) )
        self.all_params.append(self.dist_scale)
        # get instructions for how to merge bottom-up and top-down info
        self.merge_info = merge_info
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

    def apply_td(self, rand_vals):
        """
        Apply this generator network using the given random values.
        """
        assert (len(rand_vals) == len(self.td_modules)), \
                "random values should be appropriate for this network."
        assert not (rand_vals[0] is None), \
                "need explicit rand_vals[0]."
        td_acts = []
        td_pre_acts = []
        for i, (rvs, td_module) in enumerate(zip(rand_vals, self.td_modules)):
            td_mod_name = td_module.mod_name
            if td_mod_name in self.merge_info:
                # handle computation for a TD module whose output will be
                # perturbed stochastically.
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                im_module = self.im_module_dict[im_mod_name]
                if i == 0:
                    # top module in network, the TD module is just a dummy
                    # module here... no need to apply it. Conceptually, we can
                    # think of this as a TD module whose output is 0 everywhere
                    # and the IM module transforms it as described below.
                    td_act = im_module.apply_td(td_pre_act=None,
                                                rand_vals=rvs)
                    td_pre_act = 0.0 * td_act
                else:
                    # internal module in network, need to apply the TD module
                    # to get a TD pre-activation, then perturb it using the
                    # IM module (and apply, e.g. relu) to get a TD activation.
                    _, td_pre_act = td_module.apply(input=td_acts[-1])
                    td_act = im_module.apply_td(td_pre_act=td_pre_act,
                                                rand_vals=rvs)
            else:
                # handle computation for a TD module whose output isn't
                # perturbed stochastically. These TD modules receive an
                # activation as input, and produce an activation as output.
                td_act, td_pre_act = td_module.apply(input=td_acts[-1])
            td_acts.append(td_act)
            td_pre_acts.append(td_pre_act)
        # package results into a nice dict
        td_res_dict = {}
        td_res_dict['td_acts'] = td_acts
        td_res_dict['td_pre_acts'] = td_pre_acts
        return result

    def apply_bu(self, input):
        """
        Apply this model's bottom-up inference modules to the given input,
        and return a dict mapping BU module names to their outputs.
        """
        bu_acts = [input]
        bu_pre_acts = [input]
        bu_mod_res = {}
        for i, bu_mod in enumerate(self.bu_modules):
            # apply BU module
            bu_act, bu_pre_act = bu_mod.apply(input=bu_acts[i])
            # collect results
            bu_acts.append(bu_act)
            bu_pre_acts.append(bu_pre_act)
            bu_mod_res[bu_mod.mod_name] = {'bu_act': bu_act,
                                           'bu_pre_act': bu_pre_act}
        # package results into a handy dict
        bu_res_dict = {}
        bu_res_dict['bu_acts'] = bu_acts
        bu_res_dict['bu_pre_acts'] = bu_pre_acts
        bu_res_dict['bu_mod_res'] = bu_mod_res
        return bu_res_dict

    def apply_im(self, input):
        """
        Compute the merged pass over this model's bottom-up, top-down, and
        information merging modules.

        This first computes the full bottom-up pass to collect the output of
        each BU module.

        This then computes the top-down pass using latent variables sampled
        from distributions determined by merging partial results of the BU pass
        with results from the partially-completed TD pass.
        """
        # set aside a dict for recording KLd info at each layer where we use
        # conditional distributions over the latent variables.
        kld_dict = {}
        logz_dict = {'log_p_z': [], 'log_q_z': []}
        # first, run the bottom-up pass
        bu_res_dict = self.apply_bu(input)
        bu_mod_res = bu_res_dict['bu_mod_res']
        # now, run top-down pass using latent variables sampled from
        # conditional distributions constructed by merging bottom-up and
        # top-down information.
        td_acts = []
        td_pre_acts = []
        for i, td_module in enumerate(self.td_modules):
            td_mod_name = td_module.mod_name
            if td_mod_name in self.merge_info:
                # handle computation for a TD module whose output will be
                # perturbed stochastically.
                #
                # get the BU and IM modules required by this TD module
                im_mod_name = self.merge_info[td_mod_name]['im_module']
                bu_mod_name = self.merge_info[td_mod_name]['bu_module']
                im_module = self.im_module_dict[im_mod_name]
                bu_module = self.bu_module_dict[bu_mod_name]
                # get the BU pre-activations that we'll merge with this TD
                # module's pre-activations.
                bu_pre_act = bu_mod_res[bu_mod_name]['pre_act']
                if i == 0:
                    # no TD module to provide pre-activations, so we go
                    # straight to the IM module
                    im_res_dict = im_module.apply_im(td_pre_act=None,
                                                     bu_pre_act=bu_pre_act,
                                                     dist_scale=self.dist_scale)
                    # make some dummy pre-activations, for symmetry
                    td_pre_act = 0.0 * im_res_dict['td_act']
                else:
                    # propagate through this TD module to get pre-activations
                    _, td_pre_act = td_module.apply(input=td_acts[-1])
                    # go through the IM module, to get the final TD activations
                    im_res_dict = im_module.apply_im(td_pre_act=td_pre_act,
                                                     bu_pre_act=bu_pre_act,
                                                     dist_scale=self.dist_scale)
                # record TD activations produced by current TD/IM pair
                td_acts.append(im_res_dict['td_act'])
                td_pre_acts.append(td_pre_act)
                # record KLd cost from inside the IM module
                kld_dict[td_mod_name] = im_res_dict['kld']
                # record log probability of z under p and q, for IWAE bound
                logz_dict['log_p_z'].append(im_res_dict['log_p_z'])
                logz_dict['log_q_z'].append(im_res_dict['log_q_z'])
            else:
                # handle computation for a TD module whose output isn't
                # perturbed stochastically. These TD modules receive an
                # activation as input, and produce an activation as output.
                td_act, td_pre_act = td_module.apply(input=td_acts[-1])
                td_acts.append(td_act_i)
                td_pre_acts.append(td_pre_act)
        # package results into a nice dict
        im_res_dict = {}
        im_res_dict['td_acts'] = td_acts
        im_res_dict['td_pre_acts'] = td_pre_acts
        im_res_dict['bu_acts'] = bu_res_dict['bu_acts']
        im_res_dict['bu_pre_acts'] = bu_res_dict['bu_pre_acts']
        im_res_dict['kld_dict'] = kld_dict
        im_res_dict['log_p_z'] = logz_dict['log_p_z']
        im_res_dict['log_q_z'] = logz_dict['log_q_z']
        return im_res_dict
















##############
# EYE BUFFER #
##############
