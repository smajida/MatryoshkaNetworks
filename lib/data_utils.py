import numpy as np
import numpy.random as npr
from sklearn import utils as skutils
import theano

from rng import np_rng, py_rng


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


def sample_patch_masks(X, im_shape, patch_shape, patch_count=1):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rows = patch_shape[0]
    cols = patch_shape[1]
    off_row = npr.randint(1, high=(im_shape[0] - rows - 1),
                          size=(obs_count, patch_count))
    off_col = npr.randint(1, high=(im_shape[1] - cols - 1),
                          size=(obs_count, patch_count))
    dummy = np.zeros(im_shape)
    mask = np.ones(X.shape)
    for i in range(obs_count):
        for j in range(patch_count):
            # reset the dummy mask
            dummy = (0.0 * dummy) + 1.0
            # select a random patch in the dummy mask
            dummy[off_row[i, j]:(off_row[i, j] + rows), off_col[i, j]:(off_col[i, j] + cols)] = 0.0
            # turn off the patch in the final mask
            mask[i, :] = mask[i, :] * dummy.ravel()
    return mask.astype(theano.config.floatX)


def get_masked_data(xi,
                    drop_prob=0.0,
                    occ_dim=None,
                    occ_count=1,
                    data_mean=None):
    """
    Construct randomly masked data from xi.
    """
    if data_mean is None:
        data_mean = np.zeros((xi.shape[1],))
    im_dim = int(xi.shape[1]**0.5)  # images should be square
    xo = xi.copy()
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_dim is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = \
            sample_patch_masks(xi,
                               (im_dim, im_dim),
                               (occ_dim, occ_dim),
                               patch_count=occ_count)
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm


def sample_data_masks(xi, drop_prob=0.0, occ_dim=None, occ_count=1):
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
    if occ_dim is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = \
            sample_patch_masks(xi,
                               (im_dim, im_dim),
                               (occ_dim, occ_dim),
                               patch_count=occ_count)
    xm = xm_rand * xm_patch
    xm = to_fX(xm)
    return xm


# def construct_autoreg_data(xi,
#                            occ_dim=None,
#                            occ_count=1,
#                            data_mean=None):
#     '''
#     Construct randomly masked data from xi.
#     '''
#     if data_mean is None:
#         data_mean = np.zeros((xi.shape[1],))
#     im_dim = int(xi.shape[1]**0.5)  # images should be square
#     xo = xi.copy()
#     if occ_dim is None:
#         # don't apply rectangular occlusion
#         xm_patch = np.ones(xi.shape)
#     else:
#         # apply rectangular occlusion
#         xm_patch = \
#             sample_patch_masks(xi,
#                                (im_dim, im_dim),
#                                (occ_dim, occ_dim),
#                                patch_count=occ_count)
#     xm = xm_patch
#     xi = (xm * xi) + ((1.0 - xm) * data_mean)
#     xi = to_fX(xi)
#     xo = to_fX(xo)
#     xm = to_fX(xm)
#     return xi, xo, xm


def get_downsampling_masks(
        xi,
        im_shape,
        im_chans=1,
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
    img_mask = np.zeros((im_chans, rows, cols))
    for r in rows:
        for c in cols:
            if (r % 2 == 0) and (c % 2 == 0):
                img_mask[:, r, c] = 1.
    img_mask = img_mask.flatten()[np.newaxis, :]
    xm = np.repeat(img_mask, xi.shape[0], axis=0)
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm


def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X
