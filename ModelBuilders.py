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
    GenConvGRUModule, InfConvMergeModuleIMS
from MatryoshkaNetworks import InfGenModel, CondInfGenModel


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


def build_mnist_conv_res(nz0=32, nz1=4, ngf=32, ngfc=128, use_bn=False,
                         act_func='lrelu', use_td_cond=True,
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

    # construct the "wrapper" object for managing all our modules
    def noop(x):
        return x

    inf_gen_model = \
        InfGenModel(
            bu_modules=bu_modules,
            td_modules=td_modules,
            im_modules=im_modules,
            sc_modules=[],
            merge_info=merge_info,
            output_transform=noop,
            use_sc=False)

    return inf_gen_model


def build_mnist_cond_res(nz0=32, nz1=4, ngf=32, ngfc=128,
                         gen_in_chans=None, inf_in_chans=None,
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
            out_chans=nc,
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
                mod_type=inf_mt,
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
                mod_type=inf_mt,
                act_func=act_func,
                mod_name=mod_name)
        im_modules_14x14.append(new_module)

    im_modules = [im_module_1] + \
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
        im_modules_gen=im_modules,
        bu_modules_inf=bu_modules_inf,
        im_modules_inf=im_modules,
        merge_info=merge_info,
        output_transform=output_noop)

    return inf_gen_model







##############
# EYE BUFFER #
##############
