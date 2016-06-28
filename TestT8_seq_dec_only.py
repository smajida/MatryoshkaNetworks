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
desc = 'test_seq_dec_only_512d'
result_dir = '{}/results/{}'.format(EXP_DIR, desc)
inf_gen_param_file = '{}/inf_gen_params.pkl'.format(result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load text8 character sequence
data_path = '{}/data'.format(EXP_DIR)
char_seq, idx2char, char2idx = load_text8(data_path)

set_seed(123)       # seed for shared rngs
nc = len(idx2char)  # # of possible chars
ns = 100            # length of input sequences
ng = 10             # length of occluded gaps
nbatch = 50         # # of examples in batch
nz0 = 64            # # of dim for Z0
nz1 = 8             # # of dim for Z1
ngf = 512           # dimension of GRU state
niter = 500         # # of iter at starting learning rate
niter_decay = 500   # # of iter to linearly decay learning rate to zero
padding = 4         # padding to keep gaps away from sequence edges


def train_transform(X):
    # transform arrays (nbatch, ns, nc) -> (nbatch, nc, ns, 1)
    return floatX(X.reshape(-1, ns, nc, 1).transpose(0, 2, 1, 3))


tanh = activations.Tanh()
sigmoid = activations.Sigmoid()
bce = T.nnet.binary_crossentropy


########################################
# Build the sequential decoding layer. #
########################################

seq_decoder = \
    ContextualGRU(
        state_chans=ngf,
        input_chans=nc,
        output_chans=nc,
        context_chans=nc,
        act_func='tanh',
        mod_name='seq_dec')
# ...
all_params = seq_decoder.params


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


# parameter regularization part of cost
reg_cost = 1e-5 * sum([T.sum(p**2.0) for p in all_params])

seq_Xg_inf = Xg_inf.dimshuffle(0, 2, 1, 3)
seq_Xg_inf = T.flatten(seq_Xg_inf, 3)
seq_Xm_inf = Xm_inf.dimshuffle(0, 2, 1, 3)
seq_Xm_inf = T.flatten(seq_Xm_inf, 3)

# make a shifted version of char sequence to use as decoder input
dummy_vals = T.zeros((seq_Xg_inf.shape[0], 1, seq_Xg_inf.shape[2]))
seq_dec_input = T.concatenate([dummy_vals, 3. * seq_Xg_inf[:, :-1, :]], axis=1)

# run through the contextual decoder
final_preds, scan_updates = \
    seq_decoder.apply(seq_dec_input, 0. * seq_dec_input)
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
# function for applying the first step of decoder (includes state initialization)
dec_state = T.matrix()
dec_input = T.matrix()
dec_context = T.matrix()
state_1, out_0 = \
    seq_decoder.apply_step(state=None, input=dec_input, context=dec_context)
step_func_0 = theano.function([dec_input, dec_context], [state_1, out_0])
state_tp1, out_t = \
    seq_decoder.apply_step(state=dec_state, input=dec_input, context=dec_context)
step_func_1 = theano.function([dec_state, dec_input, dec_context], [state_tp1, out_t])


# function for deterministic/stochastic decoding (using argmax/sampling)
def sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False):
    xg_inf_seq = np.transpose(xg_inf, axes=(0, 2, 1, 3))
    xg_inf_seq = xg_inf_seq[:, :, :, 0]
    xm_inf_seq = np.transpose(xm_inf, axes=(0, 2, 1, 3))
    xm_inf_seq = np.mean(xm_inf_seq[:, :, :, 0], axis=2)
    # xg_inf_seq.shape: (nbatch, seq_len, char_count)
    # -- this is the ground truth character sequence
    # xm_inf_seq.shape: (nbatch, seq_len)
    # -- this is binary mask on locations to impute
    # make a shifted version of char sequence to use as decoder input
    d_vals = np.zeros((xg_inf_seq.shape[0], 1, xg_inf_seq.shape[2]))
    dec_input = np.concatenate([d_vals, 3. * xg_inf_seq[:, :-1, :]], axis=1)

    # run decoder over sequence step-by-step
    s_states = [None]                 # recurrent states
    s_outputs = [dec_input[:, 0, :]]  # recurrent predictions
    for s in range(dec_input.shape[1]):
        s_state = s_states[-1]
        s_input = 3. * s_outputs[-1]
        s_context = 0. * s_outputs[-1]
        if s == 0:
            s_out = step_func_0(s_input, s_context)
        else:
            s_out = step_func_1(s_state, s_input, s_context)
        # record the updated state
        s_states.append(s_out[0])
        # deal with sampling style for model prediction
        s_pred_true = xg_inf_seq[:, s, :]  # ground truth output for this step
        s_pred_prob = clip_softmax_np(s_out[1], axis=1)
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
        for o in range(s_pred_model.shape[0]):
            if xm_inf_seq[o, s] < 0.5:
                s_pred_model[o, :] = s_pred_true[o, :]
        # record the predictions for this step
        s_outputs.append(s_pred_model)
    # stack up sequences of predicted characters
    s_outputs = np.stack(s_outputs[1:], axis=1)
    # s_outputs.shape: (nbatch, seq_len, char_count)
    return s_outputs

# test decoder sampler
print('Testing decoder sampler...')
model_input = make_model_input(char_seq, 50)
for i, in_ary in enumerate(model_input):
    print('model_input[i].shape: {}'.format(in_ary.shape))
xg_gen, xm_gen, xg_inf, xm_inf = model_input
char_preds = sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
char_preds = sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=True)
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
        step_recons = \
            sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
        # run on fixed set of inputs
        xg_gen, xm_gen, xg_inf, xm_inf = recon_input_fixed
        step_recons_fixed = \
            sample_decoder(xg_gen, xm_gen, xg_inf, xm_inf, use_argmax=False)
        #
        # visualization for the variable set of examples
        #
        final_recons = np.argmax(step_recons, axis=2)
        rec_strs = ['********** EPOCH {} **********'.format(epoch)]
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
        for j in range(final_recons.shape[0]):
            rec_str = [idx2char[idx] for idx in final_recons[j]]
            rec_strs.append(''.join(rec_str))
        joint_str = '\n'.join(rec_strs)
        recon_fixed_out_file.write(joint_str + '\n')
        recon_fixed_out_file.flush()





##############
# EYE BUFFER #
##############
