import os
from time import time
import numpy as np
import numpy.random as npr
from tqdm import tqdm

import sys

import theano
import theano.tensor as T

#
# DCGAN paper repo stuff
#
from lib import activations
from lib import updates
from lib import inits
from lib.ops import log_mean_exp, binarize_data
from lib.costs import log_prob_categorical
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed
from lib.theano_utils import floatX, sharedX
from lib.data_utils import \
    shuffle, iter_data, sample_onehot_subseqs, get_masked_seqs
from load import load_text8

#
# Phil's business
#
from MatryoshkaModulesRNN import \
    TDModuleWrapperRNN, BUModuleWrapperRNN, \
    GenConvGRUModuleRNN, BasicConvModuleNEW, \
    InfConvGRUModuleRNN, BasicConvGRUModuleRNN, \
    ContextualGRU
from MatryoshkaNetworksRNN import DeepSeqCondGenRNN

sys.setrecursionlimit(100000)

#
# Whoa!, What's happening?
#

# path for dumping experiment info and fetching dataset
EXP_DIR = './text8'

# setup paths for dumping diagnostic info
desc = 'test_1d_rnn_seq_dec_6_steps_no_shortcut'
result_dir = '{}/results/{}'.format(EXP_DIR, desc)
inf_gen_param_file = '{}/inf_gen_params.pkl'.format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load text8 character sequence
data_path = '{}/data'.format(EXP_DIR)
char_seq, idx2char, char2idx = load_text8(data_path)

set_seed(123)       # seed for shared rngs
nc = len(idx2char)  # # of possible chars
ns = 64             # length of input sequences
ng = 8              # length of occluded gaps
nbatch = 50         # # of examples in batch
nz0 = 64            # # of dim for Z0
nz1 = 8             # # of dim for Z1
ngf = 80            # base # of channels for defining layers
ngc = 128           # dimension of "context" to feed into sequential decoder
niter = 500         # # of iter at starting learning rate
niter_decay = 500   # # of iter to linearly decay learning rate to zero
bu_act_func = 'lrelu'  # activation function for bottom-up modules
use_td_cond = True
recon_steps = 6
use_rand = True
seq_dec_shortcut = False


def train_transform(X):
    # transform arrays (nbatch, ns, nc) -> (nbatch, nc, ns, 1)
    return floatX(X.reshape(-1, ns, nc, 1).transpose(0, 2, 1, 3))


tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy

#########################################
# Setup the top-down processing modules #
# -- these are for generator            #
#########################################

td_module_1a = \
    GenConvGRUModuleRNN(
        state_chans=(ngf * 4),
        input_chans=(2 * ngf * 4),
        rand_chans=nz1,
        spatial_shape=((ns // 4), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='td_mod_1a')
td_module_1b = \
    BasicConvModuleNEW(
        in_chans=(ngf * 4),
        out_chans=(ngf * 4),
        filt_shape=(5, 1),
        stride='half',
        is_1d=True,
        act_func='ident',
        mod_name='td_mod_1b')
td_module_1 = \
    TDModuleWrapperRNN(
        td_module=td_module_1a,
        mlp_modules=[td_module_1b],
        use_rand=use_rand,
        mod_name='td_mod_1')

td_module_2a = \
    GenConvGRUModuleRNN(
        state_chans=(ngf * 4),
        input_chans=(2 * ngf * 4),
        rand_chans=nz1,
        spatial_shape=((ns // 2), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='td_mod_2a')
td_module_2b = \
    BasicConvModuleNEW(
        in_chans=(ngf * 4),
        out_chans=(ngf * 2),
        filt_shape=(5, 1),
        stride='half',
        is_1d=True,
        act_func='ident',
        mod_name='td_mod_2b')
td_module_2 = \
    TDModuleWrapperRNN(
        td_module=td_module_2a,
        mlp_modules=[td_module_2b],
        use_rand=use_rand,
        mod_name='td_mod_2')

td_module_3a = \
    GenConvGRUModuleRNN(
        state_chans=(ngf * 2),
        input_chans=(2 * ngf * 2),
        rand_chans=nz1,
        spatial_shape=((ns // 1), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='td_mod_3a')
td_module_3b = \
    BasicConvModuleNEW(
        in_chans=(ngf * 2),
        out_chans=(ngc * 2),
        filt_shape=(3, 1),
        stride='single',
        is_1d=True,
        act_func='ident',
        mod_name='td_mod_3b')
td_module_3 = \
    TDModuleWrapperRNN(
        td_module=td_module_3a,
        mlp_modules=[td_module_3b],
        use_rand=use_rand,
        mod_name='td_mod_3')

td_modules = [td_module_1, td_module_2, td_module_3]

###########################################
# Setup some bottom-up processing modules #
# -- these are for the generator          #
###########################################

bu_module_1a = \
    BasicConvGRUModuleRNN(
        state_chans=(ngf * 4),
        input_chans=(ngf * 4),
        spatial_shape=((ns // 4), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='bu_mod_1a')
bu_module_1b = \
    BasicConvModuleNEW(
        in_chans=(ngf * 4),
        out_chans=(ngf * 4),
        filt_shape=(5, 1),
        stride='double',
        is_1d=True,
        act_func='ident',
        mod_name='bu_mod_1b')
bu_module_1 = \
    BUModuleWrapperRNN(
        bu_module=bu_module_1a,
        mlp_modules=[bu_module_1b],
        mod_name='bu_mod_1')

bu_module_2a = \
    BasicConvGRUModuleRNN(
        state_chans=(ngf * 4),
        input_chans=(ngf * 4),
        spatial_shape=((ns // 2), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='bu_mod_2a')
bu_module_2b = \
    BasicConvModuleNEW(
        in_chans=(ngf * 2),
        out_chans=(ngf * 4),
        filt_shape=(5, 1),
        stride='double',
        is_1d=True,
        act_func='ident',
        mod_name='bu_mod_2b')
bu_module_2 = \
    BUModuleWrapperRNN(
        bu_module=bu_module_2a,
        mlp_modules=[bu_module_2b],
        mod_name='bu_mod_2')

bu_module_3a = \
    BasicConvGRUModuleRNN(
        state_chans=(ngf * 2),
        input_chans=(ngf * 2),
        spatial_shape=((ns // 1), 1),
        filt_shape=(3, 1),
        is_1d=True,
        act_func='tanh',
        mod_name='bu_mod_3a')
bu_module_3b = \
    BasicConvModuleNEW(
        in_chans=(2 * nc + ngc),
        out_chans=(ngf * 2),
        filt_shape=(3, 1),
        stride='single',
        is_1d=True,
        act_func='ident',
        mod_name='bu_mod_3b')
bu_module_3 = \
    BUModuleWrapperRNN(
        bu_module=bu_module_3a,
        mlp_modules=[bu_module_3b],
        mod_name='bu_mod_3')

# modules must be listed in "evaluation order"
bu_modules = [bu_module_3, bu_module_2, bu_module_1]


###########################################
# Setup some bottom-up processing modules #
# -- these are for the inferencer         #
###########################################

bu_module_1 = \
    BasicConvModuleNEW(
        in_chans=(ngf * 4),
        out_chans=(ngf * 4),
        filt_shape=(5, 1),
        stride='double',
        is_1d=True,
        act_func=bu_act_func,
        mod_name='bu_mod_1')

bu_module_2 = \
    BasicConvModuleNEW(
        in_chans=(ngf * 2),
        out_chans=(ngf * 4),
        filt_shape=(5, 1),
        stride='double',
        is_1d=True,
        act_func=bu_act_func,
        mod_name='bu_mod_2')

bu_module_3 = \
    BasicConvModuleNEW(
        in_chans=(3 * nc + ngc),
        out_chans=(ngf * 2),
        filt_shape=(5, 1),
        stride='single',
        is_1d=True,
        act_func=bu_act_func,
        mod_name='bu_mod_3')

# modules must be listed in "evaluation order"
bu_modules_inf = [bu_module_3, bu_module_2, bu_module_1]

#########################################
# Setup the information merging modules #
#########################################

im_module_1 = \
    InfConvGRUModuleRNN(
        state_chans=(ngf * 4),
        td_state_chans=(ngf * 4),
        td_input_chans=(2 * ngf * 4),
        bu_chans=(ngf * 4),
        rand_chans=nz1,
        spatial_shape=((ns // 4), 1),
        is_1d=True,
        act_func='tanh',
        use_td_cond=use_td_cond,
        mod_name='im_mod_1')

im_module_2 = \
    InfConvGRUModuleRNN(
        state_chans=(ngf * 4),
        td_state_chans=(ngf * 4),
        td_input_chans=(2 * ngf * 4),
        bu_chans=(ngf * 4),
        rand_chans=nz1,
        spatial_shape=((ns // 2), 1),
        is_1d=True,
        act_func='tanh',
        use_td_cond=use_td_cond,
        mod_name='im_mod_2')

im_module_3 = \
    InfConvGRUModuleRNN(
        state_chans=(ngf * 2),
        td_state_chans=(ngf * 2),
        td_input_chans=(2 * ngf * 2),
        bu_chans=(ngf * 2),
        rand_chans=nz1,
        spatial_shape=((ns // 1), 1),
        is_1d=True,
        act_func='tanh',
        use_td_cond=use_td_cond,
        mod_name='im_mod_3')

im_modules = [im_module_1, im_module_2, im_module_3]


#
# Setup a description for where to get conditional distributions from.
#
merge_info = {
    'td_mod_1': {'td_type': 'top', 'im_module': 'im_mod_1',
                 'bu_module': 'bu_mod_1'},
    'td_mod_2': {'td_type': 'cond', 'im_module': 'im_mod_2',
                 'bu_module': 'bu_mod_2'},
    'td_mod_3': {'td_type': 'cond', 'im_module': 'im_mod_3',
                 'bu_module': 'bu_mod_3'}
}


# construct the "wrapper" object for managing all our modules
seq_cond_gen_model = \
    DeepSeqCondGenRNN(
        td_modules=td_modules,
        bu_modules=bu_modules,
        im_modules=im_modules,
        bu_modules_inf=bu_modules_inf,
        merge_info=merge_info)

########################################
# Build the sequential decoding layer. #
########################################

seq_decoder = \
    ContextualGRU(
        state_chans=(ngc * 4),
        input_chans=nc,
        output_chans=nc,
        context_chans=ngc,
        use_shortcut=seq_dec_shortcut,
        act_func='tanh',
        mod_name='seq_dec')

####################################
# Setup the optimization objective #
####################################
lam_kld = sharedX(floatX([1.0]))
c0 = sharedX(floatX(np.zeros((ngc,))))  # initial context
x0 = sharedX(floatX(np.zeros((nc,))))   # gap filler
gen_params = seq_cond_gen_model.gen_params + seq_decoder.params + [c0, x0]
inf_params = seq_cond_gen_model.inf_params
all_params = seq_cond_gen_model.all_params + seq_decoder.params + [c0, x0]


######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################


def make_model_input(source_seq, batch_size):
    # sample a batch of one-hot subsequences from the source sequence
    x_in = sample_onehot_subseqs(source_seq, batch_size, ns, nc)
    # sample masked versions of each sequence in the minibatch
    xg_gen, xg_inf, xm_gen = \
        get_masked_seqs(x_in, drop_prob=0.0, occ_len=ng,
                        occ_count=1, data_mean=None)
    # for each x, x.shape = (nbatch, ns, nc)

    # reshape and process data for use as model input
    xm_gen = 1. - xm_gen  # mask is 1 for unobserved pixels
    xm_inf = xm_gen       # mask is 1 for pixels to predict
    # transform arrays (nbatch, ns, nc) -> (nbatch, nc, ns, 1)
    xg_gen = train_transform(xg_gen)
    xm_gen = train_transform(xm_gen)
    xg_inf = train_transform(xg_inf)
    xm_inf = train_transform(xm_inf)
    return xg_gen, xm_gen, xg_inf, xm_inf


def clip_softmax(x, axis=2):
    x = T.clip(x, -10., 10.)
    x = x - T.max(x, axis=axis, keepdims=True)
    x = T.exp(x)
    x = x / T.sum(x, axis=axis, keepdims=True)
    return x


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -10.0, 10.0))
    return output


######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################

# Setup symbolic vars for the model inputs, outputs, and costs
Xg_gen = T.tensor4()  # input to generator, with some parts masked out
Xm_gen = T.tensor4()  # mask indicating parts that are masked out
Xg_inf = T.tensor4()  # complete observation, for input to inference net
Xm_inf = T.tensor4()  # mask for which bits to predict
# get the full inputs to the generator and inferencer networks
Xa_gen = T.concatenate([Xg_gen, Xm_gen], axis=1)
Xa_inf = T.concatenate([Xg_gen, Xm_gen, Xg_inf, Xm_inf], axis=1)


##########################################################
# CONSTRUCT COST VARIABLES FOR THE VAE PART OF OBJECTIVE #
##########################################################
from theano.printing import Print

# parameter regularization part of cost
vae_reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in all_params])

# feed all masked inputs through the inference network
td_states = None
bu_states = None
im_states = None
n_batch = Xg_gen.shape[0]
seq_len = Xg_gen.shape[2]
context = c0.dimshuffle('x', 0, 'x', 'x')
context = T.repeat(context, n_batch, axis=0)
context = T.repeat(context, seq_len, axis=2)
filler = x0.dimshuffle('x', 0, 'x', 'x')
filler = T.repeat(filler, n_batch, axis=0)
filler = clip_softmax(T.repeat(filler, seq_len, axis=2), axis=1)
Xg_in = ((1. - Xm_gen) * Xg_gen) + (Xm_gen * filler)
kld_dicts = []
step_context = []
# generate the context information for the sequential decoder
for i in range(recon_steps):
    # mix observed input and current working state to make input
    # for the next step of refinement
    step_context.append(context)
    # concatenate all inputs to generator and inferencer
    Xa_gen_i = T.concatenate([Xg_in, Xm_gen, context], axis=1)
    Xa_inf_i = T.concatenate([Xg_in, Xm_gen, Xg_inf, context], axis=1)
    # run a guided refinement step
    res_dict = \
        seq_cond_gen_model.apply_im_cond(
            input_gen=Xa_gen_i,
            input_inf=Xa_inf_i,
            td_states=td_states,
            bu_states=bu_states,
            im_states=im_states)
    output_2d = res_dict['output']
    out_cont = output_2d[:, :ngc, :, :]
    out_gate = output_2d[:, ngc:, :, :]
    # perform a gated update of the decoder context
    context = (clip_sigmoid(1. + out_gate) * context) + out_cont
    # grab updated states for next refinement step
    td_states = res_dict['td_states']
    bu_states = res_dict['bu_states']
    im_states = res_dict['im_states']
    # record klds from this step
    kld_dicts.append(res_dict['kld_dict'])
# record final context
step_context.append(context)
# shuffle dims to get scan inputs.
# -- want shape: (nbatch, seq_len, chans)
# -- have shape: (nbatch, chans, seq_len, 1)
seq_context = context.dimshuffle(0, 2, 1, 3)
seq_Xg_inf = Xg_inf.dimshuffle(0, 2, 1, 3)
seq_context = T.flatten(seq_context, 3)
seq_Xg_inf = T.flatten(seq_Xg_inf, 3)

# run through the contextual decoder
final_preds, scan_updates = \
    seq_decoder.apply(seq_Xg_inf, seq_context)
# final_preds comes from scan with shape: (nbatch, seq_len, nc)
# -- need to shape it back to (nbatch, nc, seq_len, 1)
final_preds = final_preds.dimshuffle(0, 2, 1, 'x')
final_preds = clip_softmax(final_preds, axis=1)
# -- predictions come back in final_preds
Xg_recon = ((1. - Xm_gen) * Xg_gen) + (Xm_gen * final_preds)

# compute masked reconstruction error from final step.
log_p_x = T.sum(log_prob_categorical(
                T.flatten(Xg_inf, 2), T.flatten(final_preds, 2),
                mask=T.flatten(Xm_inf, 2), do_sum=False),
                axis=1)

# compute reconstruction error part of free-energy
vae_obs_nlls = -1.0 * log_p_x
vae_nll_cost = T.mean(vae_obs_nlls)

# convert KL dict to aggregate KLds over inference steps
kl_by_td_mod = {tdm_name: sum([kld_dict[tdm_name] for kld_dict in kld_dicts])
                for tdm_name in kld_dicts[0].keys()}  # aggregate over refinement steps
# compute per-layer KL-divergence part of cost
kld_tuples = [(mod_name, mod_kld) for mod_name, mod_kld in kl_by_td_mod.items()]
vae_layer_klds = T.as_tensor_variable([T.mean(mod_kld) for mod_name, mod_kld in kld_tuples])
vae_layer_names = [mod_name for mod_name, mod_kld in kld_tuples]

# compute total per-observation KL-divergence part of cost
vae_obs_klds = sum([mod_kld for mod_name, mod_kld in kld_tuples])
vae_kld_cost = T.mean(vae_obs_klds)

# compute the KLd cost to use for optimization
opt_kld_cost = lam_kld[0] * vae_kld_cost

# combined cost for generator stuff
vae_cost = vae_nll_cost + vae_kld_cost
vae_obs_costs = vae_obs_nlls + vae_obs_klds
# cost used by the optimizer
full_cost = vae_nll_cost + opt_kld_cost + vae_reg_cost


#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

print('Compiling test function...')
test_outputs = [full_cost, Xg_inf, Xm_inf, final_preds]
test_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], test_outputs)
model_input = make_model_input(char_seq, 50)
for i, in_ary in enumerate(model_input):
    print('model_input[i].shape: {}'.format(in_ary.shape))

test_out = test_func(*model_input)
to_full_cost = test_out[0]
to_Xg_inf = test_out[1]
to_Xm_inf = test_out[2]
to_final_preds = test_out[3]
print('full_cost: {}'.format(to_full_cost))
print('np.min(to_Xg_inf): {}'.format(np.min(to_Xg_inf)))
print('np.max(to_Xg_inf): {}'.format(np.max(to_Xg_inf)))
print('np.min(to_Xm_inf): {}'.format(np.min(to_Xm_inf)))
print('np.max(to_Xm_inf): {}'.format(np.max(to_Xm_inf)))
print('np.min(to_final_preds): {}'.format(np.min(to_final_preds)))
print('np.max(to_final_preds): {}'.format(np.max(to_final_preds)))
print('DONE.')

# stuff for performing updates
lrt = sharedX(0.001)
b1t = sharedX(0.9)
updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-5, clipnorm=100.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
all_updates, all_grads = updater(all_params, full_cost, return_grads=True)
for k, v in scan_updates.items():
    all_updates[k] = v

# print("Compiling sampling and reconstruction functions...")
# recon_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], step_recons)
# model_input = make_model_input(char_seq, 50)
# test_recons = recon_func(*model_input)

print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost, full_cost, vae_cost, vae_nll_cost,
                 vae_kld_cost, vae_obs_costs, vae_layer_klds]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost', 'full_cost', 'vae_cost', 'vae_nll_cost',
              'vae_kld_cost', 'vae_obs_costs', 'vae_layer_klds']
g_cost_outputs = g_basic_costs
# compile function for computing generator costs and updates
g_train_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], g_cost_outputs, updates=all_updates)
g_eval_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], g_cost_outputs)
print "{0:.2f} seconds to compile theano functions".format(time() - t)

# make file for recording test progress
log_name = "{}/RESULTS.txt".format(result_dir)
out_file = open(log_name, 'wb')
log_name = "{}/RECONS_VAR.txt".format(result_dir)
recon_out_file = open(log_name, 'wb')
log_name = "{}/RECONS_FIX.txt".format(result_dir)
recon_fixed_out_file = open(log_name, 'wb')


print("EXPERIMENT: {}".format(desc.upper()))

# setup variables for text-based progress monitoring
recon_count = 10
recon_repeats = 3
recon_input_fixed = make_model_input(char_seq, recon_count)
recon_input_fixed = [np.repeat(ary, recon_repeats, axis=0)
                     for ary in recon_input_fixed]

# ...
n_check = 0
n_updates = 0
t = time()
kld_weights = np.linspace(0.2, 1.0, 50)
for epoch in range(1, (niter + niter_decay + 1)):
    # mess with the KLd cost
    if ((epoch - 1) < len(kld_weights)):
        lam_kld.set_value(floatX([kld_weights[epoch - 1]]))
    # lam_kld.set_value(floatX([1.0]))
    # initialize cost arrays
    g_epoch_costs = [0. for i in range(5)]
    v_epoch_costs = [0. for i in range(5)]
    epoch_layer_klds = [0. for i in range(len(vae_layer_names))]
    vae_nlls = []
    vae_klds = []
    g_batch_count = 0.
    v_batch_count = 0.
    X_dummy = np.zeros((500 * nbatch, 50))
    for imb in tqdm(iter_data(X_dummy, size=nbatch), total=500, ascii=True, ncols=80):
        # transform training batch validation batch to model input format
        imb_input = make_model_input(char_seq, nbatch)
        vmb_input = make_model_input(char_seq, nbatch)
        # train vae on training batch
        g_result = g_train_func(*imb_input)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result[:5], g_epoch_costs)]
        vae_nlls.append(1. * g_result[3])
        vae_klds.append(1. * g_result[4])
        batch_obs_costs = g_result[5]
        batch_layer_klds = g_result[6]
        epoch_layer_klds = [(v1 + v2) for v1, v2 in zip(batch_layer_klds, epoch_layer_klds)]
        g_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            v_result = g_eval_func(*vmb_input)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result[:5], v_epoch_costs)]
            v_batch_count += 1
    if (epoch == 5) or (epoch == 15) or (epoch == 30) or (epoch == 60) or (epoch == 120):
        # cut learning rate in half
        lr = lrt.get_value(borrow=False)
        lr = lr / 2.0
        lrt.set_value(floatX(lr))
        b1 = b1t.get_value(borrow=False)
        b1 = b1 + ((0.95 - b1) / 2.0)
        b1t.set_value(floatX(b1))
    if epoch > niter:
        # linearly decay learning rate
        lr = lrt.get_value(borrow=False)
        remaining_epochs = (niter + niter_decay + 1) - epoch
        lrt.set_value(floatX(lr - (lr / remaining_epochs)))
    ###################
    # SAVE PARAMETERS #
    ###################
    seq_cond_gen_model.dump_params(inf_gen_param_file)
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    epoch_layer_klds = [(c / g_batch_count) for c in epoch_layer_klds]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx[:5], g_bc_names[:5])]
    str2 = " ".join(g_bc_strs)
    nll_qtiles = np.percentile(vae_nlls, [50., 80., 90., 95.])
    str3 = "    [q50, q80, q90, q95, max](vae-nll): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        nll_qtiles[0], nll_qtiles[1], nll_qtiles[2], nll_qtiles[3], np.max(vae_nlls))
    kld_qtiles = np.percentile(vae_klds, [50., 80., 90., 95.])
    str4 = "    [q50, q80, q90, q95, max](vae-kld): {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}".format(
        kld_qtiles[0], kld_qtiles[1], kld_qtiles[2], kld_qtiles[3], np.max(vae_klds))
    kld_strs = ["{0:s}: {1:.2f},".format(ln, lk) for ln, lk in zip(vae_layer_names, epoch_layer_klds)]
    str5 = "    module kld -- {}".format(" ".join(kld_strs))
    str6 = "    validation -- nll: {0:.2f}, kld: {1:.2f}, vfe/iwae: {2:.2f}".format(
        v_epoch_costs[3], v_epoch_costs[4], v_epoch_costs[2])
    joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
#     if (epoch <= 10) or ((epoch % 10) == 0):
#         recon_input = make_model_input(char_seq, recon_count)
#         recon_input = [np.repeat(ary, recon_repeats, axis=0) for ary in recon_input]
#         seq_cond_gen_model.set_sample_switch('gen')
#         step_recons = recon_func(*recon_input)
#         step_recons_fixed = recon_func(*recon_input_fixed)
#         seq_cond_gen_model.set_sample_switch('inf')
#         #
#         # visualization for the variable set of examples
#         #
#         recons = np.vstack(step_recons)
#         grayscale_grid_vis(recons, (recon_steps + 1, recon_count * recon_repeats),
#                            "{}/recons_{}.png".format(result_dir, epoch))
#         final_recons = step_recons[-1].transpose(0, 2, 1, 3)
#         # final_recons.shape: (recon_count * recon_repeats, ns, nc, 1)
#         final_recons = np.squeeze(final_recons, axis=(3,))
#         final_recons = np.argmax(final_recons, axis=2)
#         # final_recons.shape: (recon_count, ns)
#         rec_strs = ['********** EPOCH {} **********'.format(epoch)]
#         for j in range(final_recons.shape[0]):
#             rec_str = [idx2char[idx] for idx in final_recons[j]]
#             rec_strs.append(''.join(rec_str))
#         joint_str = '\n'.join(rec_strs)
#         recon_out_file.write(joint_str + '\n')
#         recon_out_file.flush()
#         #
#         # visualization for the fixed set of examples
#         #
#         final_recons = step_recons_fixed[-1].transpose(0, 2, 1, 3)
#         # final_recons.shape: (recon_count * recon_repeats, ns, nc, 1)
#         final_recons = np.squeeze(final_recons, axis=(3,))
#         final_recons = np.argmax(final_recons, axis=2)
#         # final_recons.shape: (recon_count, ns)
#         rec_strs = ['********** EPOCH {} **********'.format(epoch)]
#         for j in range(final_recons.shape[0]):
#             rec_str = [idx2char[idx] for idx in final_recons[j]]
#             rec_strs.append(''.join(rec_str))
#         joint_str = '\n'.join(rec_strs)
#         recon_fixed_out_file.write(joint_str + '\n')
#         recon_fixed_out_file.flush()





##############
# EYE BUFFER #
##############
