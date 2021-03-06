import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from rng import t_rng, cu_rng, np_rng


def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = np_rng.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)


def flip_bits(X, flip_prob=0.1, flip_mean=0.5):
    """
    Randomize X at rate flip_prob, and set new bit with probability flip_mean.
    """
    X_shape = X.shape
    flip_mask = 1. * (np_rng.rand(*X_shape) < flip_prob)
    new_bits = 1. * (np_rng.rand(*X_shape) < flip_mean)
    X_flip = ((1. - flip_mask) * X) + (flip_mask * new_bits)
    return X_flip.astype(theano.config.floatX)


def fuzz_data(X, scale=0.1, rand_type='normal'):
    """
    Add scaled uniform or gaussian noise to X.
    """
    X_shape = X.shape
    if rand_type == 'normal':
        fuzz = scale * np_rng.randn(*X_shape)
    elif rand_type == 'uniform':
        fuzz = scale * np_rng.rand(*X_shape)
    else:
        raise Exception('unknown rand type')
    X_fuzz = X + fuzz
    return X_fuzz.astype(theano.config.floatX)


def mean_pool_rows(input, pool_count=None, pool_size=None):
    """Apply mean pooling over rows of the matrix input."""
    pooled_rows = []
    for i in xrange(pool_count):
        pool_start = i * pool_size
        pool_end = i * pool_size + pool_size
        pool_mean = T.mean(input[pool_start:pool_end, :], axis=0, keepdims=True)
        pooled_rows.append(pool_mean)
    mean_pooled_input = T.concatenate(pooled_rows, axis=0)
    return mean_pooled_input


def log_mean_exp(x, axis=None):
    assert (axis is not None), "please provide an axis along which to compute."
    m = T.max(x, axis=axis, keepdims=True)
    return m + T.log(T.mean(T.exp(x - m), axis=axis, keepdims=True))


def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x / l2norm(x, axis=axis, e=e, keepdims=keepdims)


def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)


def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d


def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x * x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y * y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist


def add_noise(x, noise=None):
    """
    Add Gauss noise to x.
    """
    if not (noise is None):
        # generate zmuv samples from the provided rng
        zmuv_gauss = cu_rng.normal(size=x.shape)
        x_noise = x + (noise[0] * zmuv_gauss)
    else:
        x_noise = x
    return x_noise


def reparametrize(z_mean, z_logvar, rng=None, rvs=None):
    """
    Gaussian reparametrization helper function.
    """
    assert not ((rng is None) and (rvs is None)), \
        "must provide either rng or rvs."
    assert ((rng is None) or (rvs is None)), \
        "must provide either rng or rvs."
    if not (rng is None):
        # generate zmuv samples from the provided rng
        zmuv_gauss = rng.normal(size=z_mean.shape)
    else:
        # use zmuv samples provided by the user
        zmuv_gauss = rvs
    reparam_gauss = z_mean + (T.exp(0.5 * z_logvar) * zmuv_gauss)
    return reparam_gauss


def reparametrize_uniform(z_mean, z_logvar, scale=0.25, rng=None, rvs=None):
    """
    Uniform, fixed KL reparametrization helper function.
    """
    assert not ((rng is None) and (rvs is None)), \
        "must provide either rng or rvs."
    assert ((rng is None) or (rvs is None)), \
        "must provide either rng or rvs."
    if not (rng is None):
        # generate uniform samples in [0, 1] from the provided rng
        u_rand = rng.uniform(size=z_mean.shape)
    else:
        # use uniform samples in [0, 1] provided by the user
        u_rand = rvs
    # compute the fixed KL divergences for this sampling distribution
    prob_mat = (1. / scale) * T.ones_like(T.flatten(z_mean, 2))
    kld_z = T.log(prob_mat)
    log_q_z = T.sum(kld_z, axis=1)
    log_p_z = T.zeros_like(log_q_z)
    # rescale noise to [-scale/2, scale/2] and add to z_mean, after tanh
    # bounding to be in [-(1 - scale/2), (1 - scale/2)]
    u_rand = scale * (u_rand - 0.5)
    z_mean = (1. - (scale / 2.)) * T.tanh(z_mean)
    z = z_mean + u_rand
    return z, kld_z, log_p_z, log_q_z


def dropout(X, p=0.):
    """
    dropout using activation scaling to avoid test time weight rescaling
    """
    if p > 0:
        retain_prob = 1 - p
        X *= cu_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def conv_cond_concat(x, y):
    """
    concatenate conditioning vector on feature map axis
    """
    return T.concatenate([x, y * T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)


def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-5, n=None):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a) * 0. + a * b_u
            b_s = (1. - a) * 1. + a * b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if not (n is None):
            # add noise in "normalized" space (i.e. prior to shift and rescale)
            X = X + (n[0] * cu_rng.normal(size=X.shape))
        if g is not None and b is not None:
            X = X * g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a) * 0. + a * u
            s = (1. - a) * 1. + a * s
        X = (X - u) / T.sqrt(s + e)
        if not (n is None):
            # add noise in "normalized" space (i.e. prior to shift and rescale)
            X = X + (n[0] * cu_rng.normal(size=X.shape))
        if g is not None and b is not None:
            X = X * g + b
    else:
        raise NotImplementedError
    return X


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w.dimshuffle(1, 0, 2, 3))
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2] * subsample[0], img.shape[3] * subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2] * subsample[0], img.shape[3] * subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img
