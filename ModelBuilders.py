import numpy as np
import numpy.random as npr
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations

#
# Phil's business
#
from MatryoshkaModules import \
    BasicConvModule, GenTopModule, InfTopModule, \
    GenConvPertModule, BasicConvPertModule, \
    GenConvGRUModule, InfConvMergeModuleIMS, ClassConvModule,\
    GMMPriorModule, TDRefinerWrapper, InfConvGRUModuleIMS
from MatryoshkaNetworks import InfGenModel, CondInfGenModel, InfGenModelGMM


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


def build_mnist_conv_res(nz0=32, nz1=4, ngf=32, ngfc=128,
                         mix_comps=0, shared_dim=None,
                         use_bn=False, act_func='lrelu', use_td_cond=True,
                         depth_7x7=5, depth_14x14=5):
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        # use act_func only at bottom of perturbation meta-module
        af_i = act_func if (i == (depth_7x7 - 1)) else 'ident'
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=af_i,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        # use act_func only at bottom of perturbation meta-module
        af_i = act_func if (i == (depth_14x14 - 1)) else 'ident'
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=af_i,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # (28, 28) -> (28, 28)
    td_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=1,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_6')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5, td_module_6])

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 2 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        # use act_func only at top of perturbation meta-module
        af_i = act_func if (i == 0) else 'ident'
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=af_i,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        # use act_func only at top of perturbation meta-module
        af_i = act_func if (i == 0) else 'ident'
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=af_i,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=1,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_6, bu_module_5]
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_6': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    if mix_comps == 0:
        # standard Gaussian prior
        inf_gen_model = \
            InfGenModel(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                sc_modules=[],
                merge_info=merge_info,
                output_transform=noop,
                use_sc=False)
    else:
        # Gaussian mixture model prior
        mix_module = GMMPriorModule(mix_comps, nz0, shared_dim=shared_dim,
                                    mod_name='gmm_prior_mod')
        inf_gen_model = \
            InfGenModelGMM(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                mix_module=mix_module,
                merge_info=merge_info,
                output_transform=noop)
    return inf_gen_model


def build_mnist_conv_res_hires(
        nz0=32, nz1=4, ngf=32, ngfc=128,
        mix_comps=0, shared_dim=None,
        use_bn=False, act_func='lrelu', use_td_cond=True,
        depth_7x7=2, depth_14x14=2, depth_28x28=2):

    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            rand_shape=(nz0,),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 7, 7),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 14, 14),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    td_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'td_mod_6{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 1),
                out_chans=(ngf * 1),
                conv_chans=(ngf * 1),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 28, 28),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_28x28.append(new_module)
    # manual stuff for parameter sharing....

    # (28, 28) -> (28, 28)
    td_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='td_mod_7')

    # (28, 28) -> (28, 28)
    td_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=1,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_8')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5])
    td_modules.extend(td_modules_28x28)
    td_modules.extend([td_module_7, td_module_8])

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 2 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    bu_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'bu_mod_6{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 1),
                out_chans=(ngf * 1),
                conv_chans=(ngf * 1),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_28x28.append(new_module)
    bu_modules_28x28.reverse()

    # (28, 28) -> (28, 28)
    bu_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_7')

    # (28, 28) -> (28, 28)
    bu_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=1,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_8')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_8, bu_module_7]
    bu_modules.extend(bu_modules_28x28)
    bu_modules.extend([bu_module_5])
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    # (14, 14) -> (28, 28)
    im_module_5 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    im_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'im_mod_6{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 1),
                bu_chans=(ngf * 1),
                im_chans=(ngf * 1),
                rand_chans=nz1,
                conv_chans=(ngf * 1),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_28x28.append(new_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)
    im_modules.extend([im_module_5])
    im_modules.extend(im_modules_28x28)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': 'im_mod_5',
                     'bu_source': None, 'im_source': im_modules_14x14[-1].mod_name},

        'td_mod_7': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_8': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_28x28):
        td_type = 'cond'
        td_mod_name = 'td_mod_6{}'.format(alphabet[i])
        im_mod_name = 'im_mod_6{}'.format(alphabet[i])
        im_src_name = 'im_mod_5'
        bu_src_name = 'bu_mod_7'
        if i > 0:
            im_src_name = 'im_mod_6{}'.format(alphabet[i - 1])
        if i < (depth_28x28 - 1):
            bu_src_name = 'bu_mod_6{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    if mix_comps == 0:
        # standard Gaussian prior
        inf_gen_model = \
            InfGenModel(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                sc_modules=[],
                merge_info=merge_info,
                output_transform=noop,
                use_sc=False)
    else:
        # Gaussian mixture model prior
        mix_module = GMMPriorModule(mix_comps, nz0, mod_name='gmm_prior_mod')
        inf_gen_model = \
            InfGenModelGMM(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                mix_module=mix_module,
                merge_info=merge_info,
                output_transform=noop)
    return inf_gen_model


def build_mnist_cond_res(nz0=32, nz1=4, ngf=32, ngfc=128,
                         gen_in_chans=None, inf_in_chans=None, out_chans=1,
                         use_bn=False, act_func='lrelu', use_td_cond=True,
                         depth_7x7=5, depth_14x14=5):
    assert ((gen_in_chans is not None) and (inf_in_chans is not None))

    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################

    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            rand_shape=(nz0,),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 7, 7),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3'
        )

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 14, 14),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # (28, 28) -> (28, 28)
    td_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=out_chans,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_6')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1] + \
                 td_modules_7x7 + \
                 [td_module_3] + \
                 td_modules_14x14 + \
                 [td_module_5, td_module_6]

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do generation inference       #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 2 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=gen_in_chans,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules_gen = [bu_module_6, bu_module_5] + \
                     bu_modules_14x14 + \
                     [bu_module_3] + \
                     bu_modules_7x7 + \
                     [bu_module_1]

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference inference        #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 2 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=inf_in_chans,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules_inf = [bu_module_6, bu_module_5] + \
                     bu_modules_14x14 + \
                     [bu_module_3] + \
                     bu_modules_7x7 + \
                     [bu_module_1]


    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules_gen = [im_module_1] + \
                     im_modules_7x7 + \
                     [im_module_3] + \
                     im_modules_14x14

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules_inf = [im_module_1] + \
                     im_modules_7x7 + \
                     [im_module_3] + \
                     im_modules_14x14

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_6': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    def output_noop(x):
        output = x
        return output

    # construct the "wrapper" object for managing all our modules
    inf_gen_model = CondInfGenModel(
        td_modules=td_modules,
        bu_modules_gen=bu_modules_gen,
        im_modules_gen=im_modules_gen,
        bu_modules_inf=bu_modules_inf,
        im_modules_inf=im_modules_inf,
        merge_info=merge_info,
        output_transform=output_noop)

    return inf_gen_model


def build_mnist_conv_res_ss(nz0=32, nz1=4, ngf=32, ngfc=128, class_count=10,
                            use_bn=True, act_func='lrelu', use_td_cond=True,
                            depth_7x7=3, depth_14x14=3):
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            aux_dim=class_count,  # number of predictions to "decode"
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # (28, 28) -> (28, 28)
    td_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=1,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_6')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5, td_module_6])

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 2 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=1,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_6, bu_module_5]
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 2, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_6': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # build a module for classification, take input from top-most 7x7 module
    cls_module = \
        ClassConvModule(
            in_chans=(ngf * 2),
            class_count=class_count,
            filt_shape=(3, 3),
            bu_source='bu_mod_2a',
            stride='single',
            mod_name='cls_module')

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    inf_gen_model = \
        InfGenModelSS(
            bu_modules=bu_modules,
            td_modules=td_modules,
            im_modules=im_modules,
            cls_module=cls_module,
            merge_info=merge_info,
            use_bu_noise=True,
            output_transform=noop)

    return inf_gen_model


def build_og_conv_res(nz0=32, nz1=4, ngf=32, ngfc=128, use_bn=False,
                      act_func='lrelu', use_td_cond=True, mix_comps=0,
                      depth_7x7=5, depth_14x14=5):
    if mix_comps > 0:
        nz1_td = nz1 + nz0
    else:
        nz1_td = nz1
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            rand_shape=(nz0,),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                rand_chans=nz1_td,
                filt_shape=(3, 3),
                rand_shape=(nz1_td, 7, 7),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1_td,
                filt_shape=(3, 3),
                rand_shape=(nz1_td, 14, 14),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # (28, 28) -> (28, 28)
    td_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=1,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_6')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5, td_module_6])

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 4 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=1,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_6, bu_module_5]
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 4),
                bu_chans=(ngf * 4),
                im_chans=(ngf * 4),
                rand_chans=nz1,
                conv_chans=(ngf * 4),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_6': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    if mix_comps == 0:
        # standard Gaussian prior
        inf_gen_model = \
            InfGenModel(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                sc_modules=[],
                merge_info=merge_info,
                output_transform=noop,
                use_sc=False)
    else:
        # Gaussian mixture model prior
        mix_module = GMMPriorModule(mix_comps, nz0, mod_name='gmm_prior_mod')
        inf_gen_model = \
            InfGenModelGMM(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                mix_module=mix_module,
                mix_everywhere=True,
                merge_info=merge_info,
                output_transform=noop)
    return inf_gen_model


def build_og_conv_res_refine(nc=1, nz0=32, nz1=4, ngf=32, ngfc=128, use_bn=False,
                             act_func='lrelu', use_td_cond=False, mix_comps=0,
                             depth_7x7=2, depth_14x14=2, depth_28x28=6):
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            rand_shape=(nz0,),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 7, 7),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1,
                filt_shape=(3, 3),
                rand_shape=(nz1, 14, 14),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func='ident',
            mod_name='td_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    td_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'td_mod_6{}'.format(alphabet[i])
        tdm_name = '{}-tdm'.format(mod_name)
        dm1_name = '{}-dm1'.format(mod_name)
        # TD module
        mod_tdm = \
            GenConvGRUModule(
                in_chans=(ngf * 1),
                out_chans=(ngf * 1),
                rand_chans=nz1,
                rand_shape=(nz1, 28, 28),
                filt_shape=(3, 3),
                use_rand=True,
                apply_bn=use_bn,
                act_func='tanh',
                mod_name=tdm_name)
        # decoder module 1
        mod_dm1 = \
            BasicConvModule(
                filt_shape=(3, 3),
                in_chans=(ngf * 1),
                out_chans=nc,
                apply_bn=False,
                stride='single',
                act_func='ident',
                mod_name=dm1_name)
        wrap_mod = \
            TDRefinerWrapper(
                gen_module=mod_tdm,
                mlp_modules=[mod_dm1],
                mod_name=mod_name)
        td_modules_28x28.append(wrap_mod)
    # share parameters among the "fine-tuning" modules
    for i in range(1, depth_28x28):
        parent_module = td_modules_28x28[0]
        child_module = td_modules_28x28[i]
        child_module.gen_module.share_params(parent_module.gen_module)
        child_module.mlp_modules[0].share_params(parent_module.mlp_modules[0])

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5])
    td_modules.extend(td_modules_28x28)

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 4 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # (28, 28) -> (28, 28)
    bu_module_6 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=nc,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_6')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_6, bu_module_5]
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 4),
                bu_chans=(ngf * 4),
                im_chans=(ngf * 4),
                rand_chans=nz1,
                conv_chans=(ngf * 4),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    # (14, 14) -> (28, 28)
    im_module_5 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func='ident',
            mod_name='im_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    im_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'im_mod_6{}'.format(alphabet[i])
        new_module = \
            InfConvGRUModuleIMS(
                td_chans=(ngf * 1),
                bu_chans=(ngf * 1),
                im_chans=(ngf * 1),
                rand_chans=nz1,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func='tanh',
                mod_name=mod_name)
        im_modules_28x28.append(new_module)
    # share parameters among the "fine-tuning" modules
    for i in range(1, depth_28x28):
        parent_module = im_modules_28x28[0]
        child_module = im_modules_28x28[i]
        child_module.share_params(parent_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)
    im_modules.extend([im_module_5])
    im_modules.extend(im_modules_28x28)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': 'im_mod_5',
                     'bu_source': None, 'im_source': im_modules_14x14[-1].mod_name}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_28x28):
        td_type = 'cond'
        td_mod_name = 'td_mod_6{}'.format(alphabet[i])
        im_mod_name = 'im_mod_6{}'.format(alphabet[i])
        im_src_name = 'im_mod_5'
        bu_src_name = 'bu_mod_6'
        if i > 0:
            im_src_name = 'im_mod_6{}'.format(alphabet[i - 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    if mix_comps == 0:
        # standard Gaussian prior
        inf_gen_model = \
            InfGenModel(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                sc_modules=[],
                merge_info=merge_info,
                output_transform=noop,
                use_sc=False)
    else:
        # Gaussian mixture model prior
        mix_module = GMMPriorModule(mix_comps, nz0, mod_name='gmm_prior_mod')
        inf_gen_model = \
            InfGenModelGMM(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                mix_module=mix_module,
                merge_info=merge_info,
                output_transform=noop)
    return inf_gen_model


def build_og_conv_res_hires(
        nz0=32, nz1=4, ngf=32, ngfc=128, mix_comps=0,
        use_bn=False, act_func='lrelu', use_td_cond=True,
        depth_7x7=2, depth_14x14=2, depth_28x28=2):
    if mix_comps > 0:
        nz1_td = nz1 + nz0
    else:
        nz1_td = nz1
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################
    # FC -> (7, 7)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            rand_shape=(nz0,),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    td_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                rand_chans=nz1_td,
                filt_shape=(3, 3),
                rand_shape=(nz1_td, 7, 7),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    td_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=nz1_td,
                filt_shape=(3, 3),
                rand_shape=(nz1_td, 14, 14),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_14x14.append(new_module)
    # manual stuff for parameter sharing....

    # (14, 14) -> (28, 28)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    td_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'td_mod_6{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 1),
                out_chans=(ngf * 1),
                conv_chans=(ngf * 1),
                rand_chans=nz1_td,
                filt_shape=(3, 3),
                rand_shape=(nz1_td, 28, 28),
                use_rand=True,
                use_conv=True,
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_28x28.append(new_module)
    # manual stuff for parameter sharing....

    # (28, 28) -> (28, 28)
    td_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='td_mod_7')

    # (28, 28) -> (28, 28)
    td_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=1,
            apply_bn=False,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_8')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1]
    td_modules.extend(td_modules_7x7)
    td_modules.extend([td_module_3])
    td_modules.extend(td_modules_14x14)
    td_modules.extend([td_module_5])
    td_modules.extend(td_modules_28x28)
    td_modules.extend([td_module_7, td_module_8])

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (7, 7) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 4 * 7 * 7),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    bu_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_7x7.append(new_module)
    bu_modules_7x7.reverse()

    # (14, 14) -> (7, 7)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    bu_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_14x14.append(new_module)
    bu_modules_14x14.reverse()

    # (28, 28) -> (14, 14)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    bu_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'bu_mod_6{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 1),
                out_chans=(ngf * 1),
                conv_chans=(ngf * 1),
                filt_shape=(3, 3),
                use_conv=True,
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_28x28.append(new_module)
    bu_modules_28x28.reverse()

    # (28, 28) -> (28, 28)
    bu_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_7')

    # (28, 28) -> (28, 28)
    bu_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=1,
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_8')

    # modules must be listed in "evaluation order"
    bu_modules = [bu_module_8, bu_module_7]
    bu_modules.extend(bu_modules_28x28)
    bu_modules.extend([bu_module_5])
    bu_modules.extend(bu_modules_14x14)
    bu_modules.extend([bu_module_3])
    bu_modules.extend(bu_modules_7x7)
    bu_modules.extend([bu_module_1])

    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (7, 7)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 4, 7, 7),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (7, 7) -> (7, 7) part of network
    im_modules_7x7 = []
    for i in range(depth_7x7):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 4),
                bu_chans=(ngf * 4),
                im_chans=(ngf * 4),
                rand_chans=nz1,
                conv_chans=(ngf * 4),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_7x7.append(new_module)

    # (7, 7) -> (14, 14)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (14, 14) -> (14, 14) part of network
    im_modules_14x14 = []
    for i in range(depth_14x14):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=nz1,
                conv_chans=(ngf * 2),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    # (14, 14) -> (28, 28)
    im_module_5 = \
        BasicConvModule(
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_5')

    # grow the (28, 28) -> (28, 28) part of network
    im_modules_28x28 = []
    for i in range(depth_28x28):
        mod_name = 'im_mod_6{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 1),
                bu_chans=(ngf * 1),
                im_chans=(ngf * 1),
                rand_chans=nz1,
                conv_chans=(ngf * 1),
                use_conv=True,
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                mod_type=0,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_28x28.append(new_module)

    im_modules = [im_module_1]
    im_modules.extend(im_modules_7x7)
    im_modules.extend([im_module_3])
    im_modules.extend(im_modules_14x14)
    im_modules.extend([im_module_5])
    im_modules.extend(im_modules_28x28)

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_7x7[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': 'im_mod_5',
                     'bu_source': None, 'im_source': im_modules_14x14[-1].mod_name},

        'td_mod_7': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_8': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_7x7):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_7x7 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_14x14):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_14x14 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_28x28):
        td_type = 'cond'
        td_mod_name = 'td_mod_6{}'.format(alphabet[i])
        im_mod_name = 'im_mod_6{}'.format(alphabet[i])
        im_src_name = 'im_mod_5'
        bu_src_name = 'bu_mod_7'
        if i > 0:
            im_src_name = 'im_mod_6{}'.format(alphabet[i - 1])
        if i < (depth_28x28 - 1):
            bu_src_name = 'bu_mod_6{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    if mix_comps == 0:
        # standard Gaussian prior
        inf_gen_model = \
            InfGenModel(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                sc_modules=[],
                merge_info=merge_info,
                output_transform=noop,
                use_sc=False)
    else:
        # Gaussian mixture model prior
        mix_module = GMMPriorModule(mix_comps, nz0, mod_name='gmm_prior_mod')
        inf_gen_model = \
            InfGenModelGMM(
                bu_modules=bu_modules,
                td_modules=td_modules,
                im_modules=im_modules,
                mix_module=mix_module,
                mix_everywhere=True,
                merge_info=merge_info,
                output_transform=noop)
    return inf_gen_model


def build_faces_cond_res(nc=3, nz0=64, nz1=4, ngf=32, ngfc=256,
                         use_bn=False, act_func='lrelu', use_td_cond=False,
                         depth_8x8=1, depth_16x16=1, depth_32x32=1):
    #########################################
    # Setup the top-down processing modules #
    # -- these do generation                #
    #########################################

    # FC -> (8, 8)
    td_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 6, 8, 8),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='td_mod_1')

    # grow the (8, 8) -> (8, 8) part of network
    td_modules_8x8 = []
    for i in range(depth_8x8):
        mod_name = 'td_mod_2{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 6),
                out_chans=(ngf * 6),
                conv_chans=(ngf * 6),
                rand_chans=(nz1 * 3),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_8x8.append(new_module)

    # (8, 8) -> (16, 16)
    td_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 6),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_3')

    # grow the (16, 16) -> (16, 16) part of network
    td_modules_16x16 = []
    for i in range(depth_16x16):
        mod_name = 'td_mod_4{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                rand_chans=(nz1 * 2),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_16x16.append(new_module)

    # (16, 16) -> (32, 32)
    td_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_5')

    # grow the (32, 32) -> (32, 32) part of network
    td_modules_32x32 = []
    for i in range(depth_32x32):
        mod_name = 'td_mod_6{}'.format(alphabet[i])
        new_module = \
            GenConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                rand_chans=(nz1 * 1),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                act_func=act_func,
                us_stride=1,
                mod_name=mod_name)
        td_modules_32x32.append(new_module)

    # (32, 32) -> (64, 64)
    td_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='td_mod_7')

    # (64, 64) -> (64, 64)
    td_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=nc,
            apply_bn=False,
            rescale_output=True,
            use_noise=False,
            stride='single',
            act_func='ident',
            mod_name='td_mod_8')

    # modules must be listed in "evaluation order"
    td_modules = [td_module_1] + \
                 td_modules_8x8 + \
                 [td_module_3] + \
                 td_modules_16x16 + \
                 [td_module_5] + \
                 td_modules_32x32 + \
                 [td_module_7, td_module_8]

    ##########################################
    # Setup the bottom-up processing modules #
    # -- these do inference                  #
    ##########################################

    # (8, 8) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 6 * 8 * 8),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (8, 8) -> (8, 8) part of network
    bu_modules_8x8 = []
    for i in range(depth_8x8):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 6),
                out_chans=(ngf * 6),
                conv_chans=(ngf * 6),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_8x8.append(new_module)
    bu_modules_8x8.reverse()

    # (16, 16) -> (8, 8)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 6),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (16, 16) -> (16, 16) part of network
    bu_modules_16x16 = []
    for i in range(depth_16x16):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_16x16.append(new_module)
    bu_modules_16x16.reverse()

    # (32, 32) -> (16, 16)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 4),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # grow the (32, 32) -> (32, 32) part of network
    bu_modules_32x32 = []
    for i in range(depth_32x32):
        mod_name = 'bu_mod_6{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_32x32.append(new_module)
    bu_modules_32x32.reverse()

    # (64, 64) -> (32, 32)
    bu_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_7')

    # (64, 64) -> (64, 64)
    bu_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(nc * 2),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_8')

    # modules must be listed in "evaluation order"
    bu_modules_gen = [bu_module_8, bu_module_7] + \
                 bu_modules_32x32 + \
                 [bu_module_5] + \
                 bu_modules_16x16 + \
                 [bu_module_3] + \
                 bu_modules_8x8 + \
                 [bu_module_1]

    # (8, 8) -> FC
    bu_module_1 = \
        InfTopModule(
            bu_chans=(ngf * 6 * 8 * 8),
            fc_chans=ngfc,
            rand_chans=nz0,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='bu_mod_1')

    # grow the (8, 8) -> (8, 8) part of network
    bu_modules_8x8 = []
    for i in range(depth_8x8):
        mod_name = 'bu_mod_2{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 6),
                out_chans=(ngf * 6),
                conv_chans=(ngf * 6),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_8x8.append(new_module)
    bu_modules_8x8.reverse()

    # (16, 16) -> (8, 8)
    bu_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 6),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_3')

    # grow the (16, 16) -> (16, 16) part of network
    bu_modules_16x16 = []
    for i in range(depth_16x16):
        mod_name = 'bu_mod_4{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 4),
                out_chans=(ngf * 4),
                conv_chans=(ngf * 4),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_16x16.append(new_module)
    bu_modules_16x16.reverse()

    # (32, 32) -> (16, 16)
    bu_module_5 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 2),
            out_chans=(ngf * 4),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_5')

    # grow the (32, 32) -> (32, 32) part of network
    bu_modules_32x32 = []
    for i in range(depth_32x32):
        mod_name = 'bu_mod_6{}'.format(alphabet[i])
        new_module = \
            BasicConvPertModule(
                in_chans=(ngf * 2),
                out_chans=(ngf * 2),
                conv_chans=(ngf * 2),
                filt_shape=(3, 3),
                apply_bn=use_bn,
                stride='single',
                act_func=act_func,
                mod_name=mod_name)
        bu_modules_32x32.append(new_module)
    bu_modules_32x32.reverse()

    # (64, 64) -> (32, 32)
    bu_module_7 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(ngf * 1),
            out_chans=(ngf * 2),
            apply_bn=use_bn,
            stride='double',
            act_func=act_func,
            mod_name='bu_mod_7')

    # (64, 64) -> (64, 64)
    bu_module_8 = \
        BasicConvModule(
            filt_shape=(3, 3),
            in_chans=(2 * (nc * 2)),
            out_chans=(ngf * 1),
            apply_bn=use_bn,
            stride='single',
            act_func=act_func,
            mod_name='bu_mod_8')

    # modules must be listed in "evaluation order"
    bu_modules_inf = [bu_module_8, bu_module_7] + \
                 bu_modules_32x32 + \
                 [bu_module_5] + \
                 bu_modules_16x16 + \
                 [bu_module_3] + \
                 bu_modules_8x8 + \
                 [bu_module_1]


    #########################################
    # Setup the information merging modules #
    #########################################

    # FC -> (8, 8)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 6, 8, 8),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (8, 8) -> (8, 8) part of network
    im_modules_8x8 = []
    for i in range(depth_8x8):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 6),
                bu_chans=(ngf * 6),
                im_chans=(ngf * 6),
                rand_chans=(nz1 * 3),
                conv_chans=(ngf * 6),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_8x8.append(new_module)

    # (8, 8) -> (16, 16)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 6),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (16, 16) -> (16, 16) part of network
    im_modules_16x16 = []
    for i in range(depth_16x16):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 4),
                bu_chans=(ngf * 4),
                im_chans=(ngf * 4),
                rand_chans=(nz1 * 2),
                conv_chans=(ngf * 4),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_16x16.append(new_module)

    # (16, 16) -> (32, 32)
    im_module_5 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_5')

    # grow the (32, 32) -> (32, 32) part of network
    im_modules_32x32 = []
    for i in range(depth_32x32):
        mod_name = 'im_mod_6{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=(nz1 * 1),
                conv_chans=(ngf * 2),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_32x32.append(new_module)

    im_modules_gen = [im_module_1] + \
                 im_modules_8x8 + \
                 [im_module_3] + \
                 im_modules_16x16 + \
                 [im_module_5] + \
                 im_modules_32x32


    # FC -> (8, 8)
    im_module_1 = \
        GenTopModule(
            rand_dim=nz0,
            out_shape=(ngf * 6, 8, 8),
            fc_dim=ngfc,
            use_fc=True,
            use_sc=False,
            apply_bn=use_bn,
            act_func=act_func,
            mod_name='im_mod_1')

    # grow the (8, 8) -> (8, 8) part of network
    im_modules_8x8 = []
    for i in range(depth_8x8):
        mod_name = 'im_mod_2{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 6),
                bu_chans=(ngf * 6),
                im_chans=(ngf * 6),
                rand_chans=(nz1 * 3),
                conv_chans=(ngf * 6),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_8x8.append(new_module)

    # (8, 8) -> (16, 16)
    im_module_3 = \
        BasicConvModule(
            in_chans=(ngf * 6),
            out_chans=(ngf * 4),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_3')

    # grow the (16, 16) -> (16, 16) part of network
    im_modules_16x16 = []
    for i in range(depth_16x16):
        mod_name = 'im_mod_4{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 4),
                bu_chans=(ngf * 4),
                im_chans=(ngf * 4),
                rand_chans=(nz1 * 2),
                conv_chans=(ngf * 4),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_16x16.append(new_module)

    # (16, 16) -> (32, 32)
    im_module_5 = \
        BasicConvModule(
            in_chans=(ngf * 4),
            out_chans=(ngf * 2),
            filt_shape=(3, 3),
            apply_bn=use_bn,
            stride='half',
            act_func=act_func,
            mod_name='im_mod_5')

    # grow the (32, 32) -> (32, 32) part of network
    im_modules_32x32 = []
    for i in range(depth_32x32):
        mod_name = 'im_mod_6{}'.format(alphabet[i])
        new_module = \
            InfConvMergeModuleIMS(
                td_chans=(ngf * 2),
                bu_chans=(ngf * 2),
                im_chans=(ngf * 2),
                rand_chans=(nz1 * 1),
                conv_chans=(ngf * 2),
                use_td_cond=use_td_cond,
                apply_bn=use_bn,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_32x32.append(new_module)

    im_modules_inf = [im_module_1] + \
                 im_modules_8x8 + \
                 [im_module_3] + \
                 im_modules_16x16 + \
                 [im_module_5] + \
                 im_modules_32x32

    #
    # Setup a description for where to get conditional distributions from.
    #
    merge_info = {
        'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                     'bu_source': 'bu_mod_1', 'im_source': None},

        'td_mod_3': {'td_type': 'pass', 'im_module': 'im_mod_3',
                     'bu_source': None, 'im_source': im_modules_8x8[-1].mod_name},

        'td_mod_5': {'td_type': 'pass', 'im_module': 'im_mod_5',
                     'bu_source': None, 'im_source': im_modules_16x16[-1].mod_name},

        'td_mod_7': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None},
        'td_mod_8': {'td_type': 'pass', 'im_module': None,
                     'bu_source': None, 'im_source': None}
    }

    # add merge_info entries for the modules with latent variables
    for i in range(depth_8x8):
        td_type = 'cond'
        td_mod_name = 'td_mod_2{}'.format(alphabet[i])
        im_mod_name = 'im_mod_2{}'.format(alphabet[i])
        im_src_name = 'im_mod_1'
        bu_src_name = 'bu_mod_3'
        if i > 0:
            im_src_name = 'im_mod_2{}'.format(alphabet[i - 1])
        if i < (depth_8x8 - 1):
            bu_src_name = 'bu_mod_2{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_16x16):
        td_type = 'cond'
        td_mod_name = 'td_mod_4{}'.format(alphabet[i])
        im_mod_name = 'im_mod_4{}'.format(alphabet[i])
        im_src_name = 'im_mod_3'
        bu_src_name = 'bu_mod_5'
        if i > 0:
            im_src_name = 'im_mod_4{}'.format(alphabet[i - 1])
        if i < (depth_16x16 - 1):
            bu_src_name = 'bu_mod_4{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }
    for i in range(depth_32x32):
        td_type = 'cond'
        td_mod_name = 'td_mod_6{}'.format(alphabet[i])
        im_mod_name = 'im_mod_6{}'.format(alphabet[i])
        im_src_name = 'im_mod_5'
        bu_src_name = 'bu_mod_7'
        if i > 0:
            im_src_name = 'im_mod_6{}'.format(alphabet[i - 1])
        if i < (depth_32x32 - 1):
            bu_src_name = 'bu_mod_6{}'.format(alphabet[i + 1])
        # add entry for this TD module
        merge_info[td_mod_name] = {
            'td_type': td_type, 'im_module': im_mod_name,
            'bu_source': bu_src_name, 'im_source': im_src_name
        }

    def output_noop(x):
        output = x
        return output

    # construct the "wrapper" object for managing all our modules
    inf_gen_model = CondInfGenModel(
        td_modules=td_modules,
        bu_modules_gen=bu_modules_gen,
        im_modules_gen=im_modules_gen,
        bu_modules_inf=bu_modules_inf,
        im_modules_inf=im_modules_inf,
        merge_info=merge_info,
        output_transform=output_noop)

    return inf_gen_model




















##############
# EYE BUFFER #
##############
