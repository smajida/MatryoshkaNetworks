import numpy as np
import numpy.random as npr
from sklearn import utils as skutils
import theano

from rng import np_rng, py_rng


def one_hot(x, n):
    '''
    convert index representation to one-hot representation
    '''
    I = np.eye(n)
    x = np.array(x)
    final_shape = tuple(list(x.shape) + [n])
    oh_ary = I[x.flatten()].reshape(final_shape)
    return oh_ary


def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph) / 2.))
    i = int(round((w - pw) / 2.))
    return x[j:(j + ph), i:(i + pw)]


def patch(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = py_rng.randint(0, (h - ph))
    i = py_rng.randint(0, (w - pw))
    x = x[j:(j + ph), i:(i + pw)]
    return x


def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)


def shuffle_simultaneously(ary_list, axis=0):
    '''
    Shuffle a list of numpy arrays simultaneously along the given axis.
    '''
    ary_len = ary_list[0].shape[axis]
    shuf_idx = np.arange(ary_len)
    npr.shuffle(shuf_idx)
    ary_list = [ary.take(shuf_idx, axis=axis) for ary in ary_list]
    return ary_list


def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


#####################################
# HELPER FUNCTIONS FOR DATA MASKING #
#####################################
def to_fX(np_ary):
    np_ary_fX = np_ary.astype(theano.config.floatX)
    return np_ary_fX


def apply_mask(Xd=None, Xc=None, Xm=None):
    """
    Apply a mask, like in the old days.
    """
    X_masked = ((1.0 - Xm) * Xd) + (Xm * Xc)
    return X_masked


def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)


def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)


def sample_patch_masks(obs_count, im_shape, patch_shape, patch_count=1):
    """
    Sample a random patch mask for each image in X.
    """
    print('im_shape: {}'.format(im_shape))
    print('patch_shape: {}'.format(patch_shape))

    rows = patch_shape[0]
    cols = patch_shape[1]
    try:
        off_row = npr.randint(1, high=(im_shape[0] - rows - 1),
                              size=(obs_count, patch_count))
    except:
        print('off_row: low=1, high={}'.format(im_shape[0] - rows - 1))
        assert False, 'derp'
    try:
        off_col = npr.randint(1, high=(im_shape[1] - cols - 1),
                              size=(obs_count, patch_count))
    except:
        print('off_col: low=1, high={}'.format(im_shape[0] - cols - 1))
        assert False, 'derp'
    dummy = np.zeros(im_shape)
    mask = np.ones((obs_count, im_shape[0] * im_shape[1]))
    for i in range(obs_count):
        for j in range(patch_count):
            # reset the dummy mask
            dummy = (0. * dummy) + 1.
            # select a random patch in the dummy mask
            dummy[off_row[i, j]:(off_row[i, j] + rows), off_col[i, j]:(off_col[i, j] + cols)] = 0.
            # turn off the patch in the final mask
            mask[i, :] = mask[i, :] * dummy.ravel()
    return mask.astype(theano.config.floatX)


def get_masked_data(xi,
                    im_shape,
                    drop_prob=0.0,
                    occ_shape=None,
                    occ_count=1,
                    data_mean=None):
    """
    Construct randomly masked data from xi.

    Assume data is passed as a 2d matrix whose rows must be reshaped
    to get either 1 channel or 3 channel images.
    """
    obs_count = xi.shape[0]
    # sample uniform random masks
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    # sample rectangular occlusion masks
    if len(im_shape) == 3:
        # 3 channel images
        im_dim = im_shape[0] * im_shape[1] * im_shape[2]
        if occ_shape is None:
            xm_patch = np.ones(xi.shape)
        else:
            # apply rectangular occlusion to 1 channel imgs
            xm_patch = \
                sample_patch_masks(obs_count,
                                   (im_shape[1], im_shape[2]),
                                   occ_shape,
                                   patch_count=occ_count)
            # expand masks to cover all channels.
            # -- assume 3 channel ims are (chans, rows, cols)
            xm_patch = xm_patch.reshape((obs_count, im_shape[1], im_shape[2]))
            xm_patch = xm_patch[:, np.newaxis, :, :]
            xm_patch = np.repeat(xm_patch, im_shape[0], axis=1)
    elif len(im_shape) == 2:
        # 1 channel images
        im_dim = im_shape[0] * im_shape[1]
        if occ_shape is None:
            # don't apply rectangular occlusion
            xm_patch = np.ones(xi.shape)
        else:
            # apply rectangular occlusion to 1 channel imgs
            xm_patch = \
                sample_patch_masks(obs_count,
                                   (im_shape[0], im_shape[1]),
                                   occ_shape,
                                   patch_count=occ_count)
    # flatten rectangular occlusion masks
    xm_patch = xm_patch.reshape(xi.shape)
    # compute data mean to swap in for masked values
    if data_mean is None:
        data_mean = np.zeros((im_dim,))
    # apply masks to images
    xo = xi.copy()
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm


def sample_data_masks(xi, drop_prob=0.0, occ_shape=None, occ_count=1):
    """
    Construct random masks for observations in xi.
    """
    im_dim = int(xi.shape[1]**0.5)  # images should be square
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_shape is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = \
            sample_patch_masks(xi,
                               (im_dim, im_dim),
                               occ_shape,
                               patch_count=occ_count)
    xm = xm_rand * xm_patch
    xm = to_fX(xm)
    return xm


def get_autoregression_masks(x_in, im_shape=(28, 28), im_chans=1,
                             order='cols', data_mean=None):
    '''
    Get masks for autoregression by progressive imputation.
    '''
    assert (order in ['cols']), \
        'unknown autoregression order: {}'.format(order)
    if data_mean is None:
        data_mean = np.zeros(x_in.shape)
    else:
        data_mean = np.repeat(data_mean[np.newaxis, :], x_in.shape[0], axis=0)
    rows = im_shape[0]
    cols = im_shape[1]
    chans = im_chans
    # init arrays to hold the generator (i.e. visibility) masks and the
    # inference (i.e. pixels to predict) masks.
    xm_gen_templates = np.ones((cols, chans, rows, cols))
    xm_inf_templates = np.zeros((cols, chans, rows, cols))
    for col in range(cols):
        # construct masks that say to predict the "first" missing column
        xm_gen_templates[col, :, :, col:] = 0.  # zero for missing pixels
        xm_inf_templates[col, :, :, col] = 1.   # one for pixels to predict
    # sample gen/inf mask pairs for the examples in x_in
    xm_gen = np.zeros(x_in.shape)
    xm_inf = np.zeros(x_in.shape)
    mask_idx = npr.randint(0, high=cols, size=(x_in.shape[0],))
    for i in range(x_in.shape[0]):
        m_idx = mask_idx[i]
        xm_gen[i, :] = xm_gen_templates[m_idx, :, :, :].flatten()
        xm_inf[i, :] = xm_inf_templates[m_idx, :, :, :].flatten()
    # construct "data" inputs to the model
    xg_gen = (xm_gen * x_in) + ((1. - xm_gen) * data_mean)
    xg_inf = x_in
    # make sure everthing's fine for the gpu
    xg_gen = to_fX(xg_gen)
    xg_inf = to_fX(xg_inf)
    xm_gen = to_fX(xm_gen)
    xm_inf = to_fX(xm_inf)
    return xg_gen, xm_gen, xg_inf, xm_inf


def get_downsampling_masks(
        xi,
        im_shape,
        im_chans=1,
        fixed_mask=True,
        data_mean=None):
    '''
    Get masked data that imitates downsampling.
    '''
    if data_mean is None:
        data_mean = np.zeros((xi.shape[1],))
    rows = im_shape[0]
    cols = im_shape[1]
    xo = xi.copy()
    # construct 2x "downsampling" mask
    img_masks = []
    doots = [(0, 0)] if fixed_mask else [(0, 0), (0, 1), (1, 0), (1, 1)]
    for doot in doots:
        img_mask = np.zeros((im_chans, rows, cols))
        for r in range(rows):
            for c in range(cols):
                if ((r + doot[0]) % 2 == 0) and \
                        ((c + doot[1]) % 2 == 0):
                    img_mask[:, r, c] = 1.
        img_mask = img_mask.flatten()
        img_masks.append(img_mask)
    xm = np.zeros(xi.shape)
    for i in range(xm.shape[0]):
        if fixed_mask:
            idx = 0
        else:
            idx = npr.randint(0, 4)
        xm[i, :] = img_masks[idx]
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm


def get_downsampled_data(
        xi,
        im_shape,
        im_chans=1,
        fixed_mask=True):
    '''
    Get masked data that imitates downsampling.
    '''
    rows = im_shape[0]
    cols = im_shape[1]
    # construct 2x "downsampling" masks
    img_masks = []
    doots = [(0, 0)] if fixed_mask else [(0, 0), (0, 1), (1, 0), (1, 1)]
    for doot in doots:
        img_mask = np.zeros((im_chans, rows, cols), dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                if ((r + doot[0]) % 2 == 0) and \
                        ((c + doot[1]) % 2 == 0):
                    img_mask[:, r, c] = 1
        img_mask = img_mask.flatten().astype(np.bool)
        img_masks.append(img_mask)
    # downsample each image in xi
    xi_ds = np.zeros((xi.shape[0], np.sum(img_masks[0])))
    for i in range(xi.shape[0]):
        if fixed_mask:
            idx = 0
        else:
            idx = npr.randint(0, 4)
        xi_ds[i, :] = xi[i, img_masks[idx]]
    return xi_ds


def sample_mnist_quadrant_masks(x_in, num_quadrants):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = x_in.shape[0]
    off_locs = [(0, 0), (0, 14), (14, 0), (14, 14)]
    dummy = np.zeros((28, 28))
    masks = np.ones((obs_count, 28 * 28))
    mask_locs = np.arange(4)
    for i in range(obs_count):
        # reset the dummy mask
        dummy = (0. * dummy) + 1.
        # get the quadrants to mask
        npr.shuffle(mask_locs)
        for j in range(num_quadrants):
            # switch off a quadrant in the dummy mask
            loc_j = off_locs[mask_locs[j]]
            o_r, o_c = loc_j
            dummy[o_r:(o_r + 14), o_c:(o_c + 14)] = 0.
        # turn off the patch in the final mask
        masks[i, :] = masks[i, :] * dummy.ravel()
    return masks.astype(theano.config.floatX)


def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X


def sample_onehot_subseqs(source_seq, seq_count, seq_len, n):
    '''
    Sample subsequences of the given source sequence, and convert to one-hot.
    '''
    source_len = source_seq.shape[0]
    max_start_idx = source_len - seq_len
    # sample the "base" sequences
    start_idx = npr.randint(low=0, high=max_start_idx, size=(seq_count,))
    idx_seqs = []
    for i in range(seq_count):
        subseq = source_seq[start_idx[i]:(start_idx[i] + seq_len)]
        idx_seqs.append(subseq[np.newaxis, :])
    idx_seqs = np.vstack(idx_seqs)
    one_hot_seqs = one_hot(idx_seqs, n=n)
    return one_hot_seqs


def get_masked_seqs(xi,
                    drop_prob=0.0,
                    occ_len=None,
                    occ_count=1,
                    data_mean=None):
    '''
    Construct randomly masked data from xi.

    Assume data is passed as 3d matrix of sequences of vectors.

    xi.shape = (nbatch, seq_len, vec_dim)
    '''
    print('get_masked_seqs(), xi.shape: {}'.format(xi.shape))
    obs_count = xi.shape[0]
    seq_shape = (xi.shape[1], xi.shape[2])
    # sample uniform random masks
    if drop_prob > 1e-3:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
        assert False, 'fully random masking not implemented yet'
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_len is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion to 1 channel imgs
        xm_patch = \
            sample_patch_masks(obs_count,
                               (seq_shape[0], seq_shape[1]),
                               (occ_len, seq_shape[1]),
                               patch_count=occ_count)
    # make default values to swap in for masked values
    if data_mean is None:
        data_mean = np.zeros(xi.shape)
    # apply masks to sequences of vectors
    xo = xi.copy()
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm


##############
# EYE BUFFER #
##############