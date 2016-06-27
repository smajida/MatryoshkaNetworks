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
ns = 64             # length of input sequences
ng = 8              # length of occluded gaps
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

# run through the contextual decoder
final_preds, scan_updates = \
    seq_decoder.apply(2. * seq_Xg_inf, 1. * seq_Xg_inf)
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
