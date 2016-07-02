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
desc = 'test_seq_dec_bd_enc_1_steps_deep'
result_dir = '{}/results/{}'.format(EXP_DIR, desc)
inf_gen_param_file = '{}/inf_gen_params.pkl'.format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load text8 character sequence
data_path = '{}/data'.format(EXP_DIR)
char_seq, idx2char, char2idx = load_text8(data_path)

set_seed(123)       # seed for shared rngs
nc = len(idx2char)  # # of possible chars
ns = 100             # length of input sequences
ng = 10             # length of occluded gaps
ngf = 512           # dimension of enc/dec GRUs
ngc = 256           # dimension of constructed context
nbatch = 50         # # of examples in batch
padding = 5         # padding to keep gap away from sequence edges
zz_steps = 1
in_scale = 3.       # rescaling factor for one-hot inputs

niter = 500         # # of iter at starting learning rate
niter_decay = 500   # # of iter to linearly decay learning rate to zero


def train_transform(X):
    # transform arrays (nbatch, ns, nc) -> (nbatch, nc, ns, 1)
    return floatX(X.reshape(-1, ns, nc, 1).transpose(0, 2, 1, 3))


tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


########################################################
# Build GRUs for the bidirectional context constructor #
########################################################

seq_enc_f = \
    ContextualGRU(
        state_chans=ngf,
        input_chans=(nc * 2),
        output_chans=(ngc * 2),
        context_chans=ngc,
        act_func='tanh',
        mod_name='seq_enc_f')

seq_enc_b = \
    ContextualGRU(
        state_chans=ngf,
        input_chans=(nc * 2),
        output_chans=(ngc * 2),
        context_chans=ngc,
        act_func='tanh',
        mod_name='seq_enc_b')

# gather encoder parameters
enc_params = seq_enc_f.params + seq_enc_b.params

########################################
# Build the sequential decoding layer. #
########################################

# first layer in deep GRU decoder
seq_decoder_1 = \
    ContextualGRU(
        state_chans=ngf,
        input_chans=nc,
        output_chans=ngf,
        context_chans=(ngc + (nc * 2)),
        act_func='tanh',
        mod_name='seq_dec')

# second layer in deep GRU decoder
seq_decoder_2 = \
    ContextualGRU(
        state_chans=ngf,
        input_chans=ngf,
        output_chans=nc,
        context_chans=ngc,
        act_func='tanh',
        mod_name='seq_dec')
# ...
dec_params = seq_decoder_1.params + seq_decoder_2.params

###################################
# Make additional trainable stuff #
###################################
c0 = sharedX(floatX(np.zeros((ngc,))))  # initial context
x0 = sharedX(floatX(np.zeros((nc,))))   # gap filler
all_params = enc_params + dec_params + [c0, x0]

######################################################
# BUILD THE MODEL TRAINING COST AND UPDATE FUNCTIONS #
######################################################


def make_model_input(source_seq, batch_size):
    # sample a batch of one-hot subsequences from the source sequence
    x_in = sample_onehot_subseqs(source_seq, batch_size, ns, nc)
    # sample masked versions of each sequence in the minibatch
    xg_gen, xg_inf, xm_gen = \
        get_masked_seqs(x_in, drop_prob=0.0, occ_len=ng,
                        occ_count=1, data_mean=None, padding=padding)
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


def clip_softmax_np(x, axis=1):
    x = np.clip(x, -10., 10.)
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=axis, keepdims=True)
    return x


def clip_sigmoid(x):
    output = sigmoid(T.clip(x, -10.0, 10.0))
    return output


def reverse_seq(seq_ary):
    seq_rev = seq_ary[:, ::-1, :]
    return seq_rev

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

# feed all masked inputs through the inference network
n_batch = Xg_gen.shape[0]
seq_len = Xg_gen.shape[2]
context = c0.dimshuffle('x', 0, 'x', 'x')
context = T.repeat(context, n_batch, axis=0)
context = T.repeat(context, seq_len, axis=2)
filler = x0.dimshuffle('x', 0, 'x', 'x')
filler = T.repeat(filler, n_batch, axis=0)
filler = clip_softmax(T.repeat(filler, seq_len, axis=2), axis=1)
Xg_in = ((1. - Xm_gen) * Xg_gen) + (Xm_gen * filler)

# parameter regularization part of cost
reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in all_params])

# make primary input sequences -- (nbatch, seq_len, feat_dim)
seq_Xg_in = Xg_in.dimshuffle(0, 2, 1, 3)
seq_Xg_in = T.flatten(seq_Xg_in, 3)
seq_X_full = Xg_inf.dimshuffle(0, 2, 1, 3)
seq_X_full = T.flatten(seq_X_full, 3)
seq_Xm_in = Xm_gen.dimshuffle(0, 2, 1, 3)
seq_Xm_in = T.flatten(seq_Xm_in, 3)
seq_context = context.dimshuffle(0, 2, 1, 3)
seq_context = T.flatten(seq_context, 3)
# seq_input is for encoder
seq_input = T.concatenate([in_scale * seq_Xg_in, seq_Xm_in], axis=2)

# run iterative zig-zag refinement
scan_updates = []
for i in range(zz_steps):
    # compute forward model's context refinement
    enc_out_f, su = \
        seq_enc_f.apply(seq_input, seq_context)
    scan_updates.append(su)
    out_ctxt = enc_out_f[:, :, :ngc]
    out_gate = clip_sigmoid(enc_out_f[:, :, ngc:] + 1.)
    # update context
    seq_context = (out_gate * seq_context) + ((1. - out_gate) * out_ctxt)
    # compute backward model's context refinement
    enc_out_b, su = \
        seq_enc_b.apply(reverse_seq(seq_input), reverse_seq(seq_context))
    scan_updates.append(su)
    enc_out_b = reverse_seq(enc_out_b)
    out_ctxt = enc_out_b[:, :, :ngc]
    out_gate = clip_sigmoid(enc_out_b[:, :, ngc:] + 1.)
    # update context
    seq_context = (out_gate * seq_context) + ((1. - out_gate) * out_ctxt)

# seq_dec_input and seq_dec_context are for decoder
dummy_vals = T.zeros((seq_X_full.shape[0], 1, seq_X_full.shape[2]))
seq_dec_input = T.concatenate([dummy_vals, in_scale * seq_X_full[:, :-1, :]], axis=1)
seq_dec_context = T.concatenate([seq_input, seq_context], axis=2)

# run through the deep, contextual GRU decoder
dec_out_1, su_1 = \
    seq_decoder_1.apply(seq_dec_input, seq_dec_context)
final_preds, su_2 = \
    seq_decoder_2.apply(dec_out_1, seq_context)

scan_updates.extend([su_1, su_2])
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
obs_nlls = -1.0 * log_p_x
nll_cost = T.mean(obs_nlls) + 1e-5 * (T.mean(Xa_gen) + T.mean(Xa_inf))

# cost used by the optimizer
full_cost = nll_cost + reg_cost


#################################################################
# COMBINE VAE AND GAN OBJECTIVES TO GET FULL TRAINING OBJECTIVE #
#################################################################

print('Compiling encoder and sampling functions...')
# function for applying the encoder and getting nicely-shaped context/input
encoder_outputs = [seq_context, seq_dec_context, seq_dec_input, full_cost]
encoder_func = theano.function([Xg_gen, Xm_gen, Xg_inf, Xm_inf], encoder_outputs)

# function for applying the first step of decoder (includes state initialization)
dec_state_1 = T.matrix()
dec_input_1 = T.matrix()
dec_context_1 = T.matrix()
dec_state_2 = T.matrix()
dec_context_2 = T.matrix()
step_init_inputs = [dec_input_1, dec_context_1, dec_context_2]
step_next_inputs = [dec_state_1, dec_input_1, dec_context_1, dec_state_2, dec_context_2]

# run the initial step through the deep GRU -- this provides initial states
# and the output for the first step
state_1_init, out_1_init = \
    seq_decoder_1.apply_step(state=None, input=dec_input_1, context=dec_context_1)
state_2_init, out_2_init = \
    seq_decoder_2.apply_step(state=None, input=out_1_init, context=dec_context_2)
# output states of both GRU layers, and the output of top layer
step_init_outputs = [state_1_init, state_2_init, out_2_init]
# compile the initial decoder step function
step_func_init = theano.function(step_init_inputs, step_init_outputs)

# run subsequent steps through the deep GRU -- this provides updated states
# and the output for each step
state_1_next, out_1_next = \
    seq_decoder_1.apply_step(state=dec_state_1, input=dec_input_1, context=dec_context_1)
state_2_next, out_2_next = \
    seq_decoder_2.apply_step(state=dec_state_2, input=out_1_next, context=dec_context_2)
# output states of both GRU layers, and the output of top layer
step_next_outputs = [state_1_next, state_2_next, out_2_next]
# compile the initial decoder step function
step_func_next = theano.function(step_next_inputs, step_next_outputs)
# NOTE: context for the decoder should be as provided by the encoder function


# function for deterministic/stochastic decoding (using argmax/sampling)
def sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False):
    enc_out = encoder_func(xg_gen, xm_gen, xg_inf, xm_inf)
    _seq_context, _seq_dec_context, _seq_dec_input = enc_out[0], enc_out[1], enc_out[2]
    # _seq_context.shape: (nbatch, seq_len, context_dim)
    # _seq_dec_context.shape: (nbatch, seq_len, context_dim)
    # _seq_dec_input.shape: (nbatch, seq_len, input_dim)

    xg_inf_seq = np.transpose(xg_inf, axes=(0, 2, 1, 3))
    xg_inf_seq = xg_inf_seq[:, :, :, 0]
    xm_inf_seq = np.transpose(xm_inf, axes=(0, 2, 1, 3))
    xm_inf_seq = np.mean(xm_inf_seq[:, :, :, 0], axis=2)
    # xg_inf_seq.shape: (nbatch, seq_len, char_count)
    # -- this is the ground truth character sequence
    # xm_inf_seq.shape: (nbatch, seq_len)
    # -- this is binary mask on locations to impute

    # run deep GRU decoder over sequence step-by-step
    s_states = [(None, None)]              # recurrent states
    s_outputs = [_seq_dec_input[:, 0, :]]  # recurrent predictions/outputs
    pred_loss = 0.                         # cumulative prediction loss (NLL)
    for s in range(_seq_dec_context.shape[1]):
        s_state_1, s_state_2 = s_states[-1]
        s_input_1 = s_outputs[-1]
        s_context_1 = _seq_dec_context[:, s, :]
        s_context_2 = _seq_context[:, s, :]
        if s == 0:
            s_out = step_func_init(s_input_1, s_context_1, s_context_2)
        else:
            s_out = step_func_next(s_state_1, s_input_1, s_context_1,
                                   s_state_2, s_context_2)
        # record the updated state
        s_states.append((s_out[0], s_out[1]))
        # deal with sampling style for model prediction
        s_pred_true = xg_inf_seq[:, s, :]  # ground truth output for this step
        s_pred_prob = clip_softmax_np(s_out[2], axis=1)
        s_pred_loss = -1. * np.sum(s_pred_true * np.log(s_pred_prob), axis=1)
        s_pred_model = np.zeros_like(s_pred_prob)
        s_roulette = np.cumsum(s_pred_prob, axis=1)
        for o in range(s_pred_model.shape[0]):
            if use_argmax:
                c_idx = np.argmax(s_pred_prob[o, :])
                s_pred_model[o, c_idx] = 1.
            else:
                r_val = npr.rand()
                for c in range(s_pred_model.shape[1]):
                    if r_val <= s_roulette[o, c]:
                        s_pred_model[o, c] = 1.
                        break
        # swap in ground truth predictions for visible parts of sequence
        s_pred_mask = np.zeros((s_pred_model.shape[0],))
        for o in range(s_pred_model.shape[0]):
            if xm_inf_seq[o, s] < 0.5:
                s_pred_model[o, :] = s_pred_true[o, :]
            else:
                s_pred_mask[o] = 1.
        # aggregate loss over imputed characters
        pred_loss = pred_loss + np.sum(s_pred_loss * s_pred_mask)
        # record the predictions for this step
        s_outputs.append(in_scale * s_pred_model)
    # convert to average loss per observation
    pred_loss = pred_loss / xg_inf_seq.shape[0]
    # stack up sequences of predicted characters
    s_outputs = np.stack(s_outputs[1:], axis=1)
    # s_outputs.shape: (nbatch, seq_len, char_count)
    return s_outputs, pred_loss

# test compiled functions
model_input = make_model_input(char_seq, 50)
for i, in_ary in enumerate(model_input):
    print('model_input[i].shape: {}'.format(in_ary.shape))
enc_out = encoder_func(*model_input)
eo_context = enc_out[0]
eo_input = enc_out[1]
print('eo_context.shape: {}'.format(eo_context.shape))
print('eo_input.shape: {}'.format(eo_input.shape))
print('DONE.')

# test decoder sampler
print('Testing decoder sampler...')
xg_gen, xm_gen, xg_inf, xm_inf = model_input
char_preds, pred_loss = sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
char_preds, pred_loss = sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=True)
print('DONE.')

# stuff for performing updates
lrt = sharedX(0.001)
b1t = sharedX(0.9)
updater = updates.Adam(lr=lrt, b1=b1t, b2=0.99, e=1e-5, clipnorm=100.0)

# build training cost and update functions
t = time()
print("Computing gradients...")
all_updates, all_grads = updater(all_params, full_cost, return_grads=True)
for scups in scan_updates:
    for k, v in scups.items():
        all_updates[k] = v

print("Compiling training functions...")
# collect costs for generator parameters
g_basic_costs = [full_cost, nll_cost, reg_cost]
g_bc_idx = range(0, len(g_basic_costs))
g_bc_names = ['full_cost', 'nll_cost', 'reg_cost']
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
for epoch in range(1, (niter + niter_decay + 1)):
    # initialize cost arrays
    g_epoch_costs = [0. for c in g_basic_costs]
    v_epoch_costs = [0. for c in g_basic_costs]
    g_batch_count = 0.
    v_batch_count = 0.
    X_dummy = np.zeros((500 * nbatch, 50))
    for imb in tqdm(iter_data(X_dummy, size=nbatch), total=500, ascii=True, ncols=80):
        # transform training batch validation batch to model input format
        imb_input = make_model_input(char_seq, nbatch)
        vmb_input = make_model_input(char_seq, nbatch)
        # train vae on training batch
        g_result = g_train_func(*imb_input)
        g_epoch_costs = [(v1 + v2) for v1, v2 in zip(g_result, g_epoch_costs)]
        g_batch_count += 1
        # evaluate vae on validation batch
        if v_batch_count < 25:
            v_result = g_eval_func(*vmb_input)
            v_epoch_costs = [(v1 + v2) for v1, v2 in zip(v_result, v_epoch_costs)]
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
    ##################################
    # QUANTITATIVE DIAGNOSTICS STUFF #
    ##################################
    g_epoch_costs = [(c / g_batch_count) for c in g_epoch_costs]
    v_epoch_costs = [(c / v_batch_count) for c in v_epoch_costs]
    str1 = "Epoch {}: ({})".format(epoch, desc.upper())
    g_bc_strs = ["{0:s}: {1:.2f},".format(c_name, g_epoch_costs[c_idx])
                 for (c_idx, c_name) in zip(g_bc_idx, g_bc_names)]
    str2 = " ".join(g_bc_strs)
    str3 = "    validation -- nll: {0:.2f}".format(v_epoch_costs[1])
    joint_str = "\n".join([str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str + "\n")
    out_file.flush()
    if (epoch <= 10) or ((epoch % 10) == 0):
        # make input for testing decoder predictions
        recon_input = make_model_input(char_seq, recon_count)
        recon_input = [np.repeat(ary, recon_repeats, axis=0) for ary in recon_input]
        # run on random set of inputs
        xg_gen, xm_gen, xg_inf, xm_inf = recon_input
        step_recons, pred_loss = \
            sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
        # run on fixed set of inputs
        xg_gen, xm_gen, xg_inf, xm_inf = recon_input_fixed
        step_recons_fixed, pred_loss_fixed = \
            sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
        #
        # visualization for the variable set of examples
        #
        final_recons = np.argmax(step_recons, axis=2)
        rec_strs = ['********** EPOCH {} **********'.format(epoch)]
        rec_strs.append('pred_loss: {0:.4f}'.format(pred_loss))
        for j in range(final_recons.shape[0]):
            rec_str = [idx2char[idx] for idx in final_recons[j]]
            rec_strs.append(''.join(rec_str))
        joint_str = '\n'.join(rec_strs)
        recon_out_file.write(joint_str + '\n')
        recon_out_file.flush()
        #
        # visualization for the fixed set of examples
        #
        final_recons = np.argmax(step_recons_fixed, axis=2)
        rec_strs = ['********** EPOCH {} **********'.format(epoch)]
        rec_strs.append('pred_loss_fixed: {0:.4f}'.format(pred_loss_fixed))
        for j in range(final_recons.shape[0]):
            rec_str = [idx2char[idx] for idx in final_recons[j]]
            rec_strs.append(''.join(rec_str))
        joint_str = '\n'.join(rec_strs)
        recon_fixed_out_file.write(joint_str + '\n')
        recon_fixed_out_file.flush()





##############
# EYE BUFFER #
##############
