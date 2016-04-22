#
# Author: Joris Vankerschaver 2013
#
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg
import theano
import theano.tensor as T


def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Elements of v smaller than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return np.array([0 if abs(x) < eps else 1/x for x in v], dtype=float)


def psd_pinv_decomposed_log_pdet(mat, cond=None, rcond=None,
                                 lower=True, check_finite=True):
    """
    Compute a decomposition of the pseudo-inverse and the logarithm of
    the pseudo-determinant of a symmetric positive semi-definite
    matrix.
    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix.
    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    cond, rcond : float or None
        Cutoff for 'small' singular values.
        Eigenvalues smaller than ``rcond*largest_eigenvalue``
        are considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `mat`. (Default: lower)
    check_finite : boolean, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    M : array_like
        The pseudo-inverse of the input matrix is np.dot(M, M.T).
    log_pdet : float
        Logarithm of the pseudo-determinant of the matrix.
    """
    # Compute the symmetric eigendecomposition.
    # The input covariance matrix is required to be real symmetric
    # and positive semidefinite which implies that its eigenvalues
    # are all real and non-negative,
    # but clip them anyway to avoid numerical issues.

    # TODO: the code to set cond/rcond is identical to that in
    # scipy.linalg.{pinvh, pinv2} and if/when this function is subsumed
    # into scipy.linalg it should probably be shared between all of
    # these routines.

    # Note that eigh takes care of array conversion, chkfinite,
    # and assertion that the matrix is square.
    s, u = scipy.linalg.eigh(mat, lower=lower, check_finite=check_finite)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(s))

    if np.min(s) < -eps:
        raise ValueError('the covariance matrix must be positive semidefinite')

    s_pinv = _pinv_1d(s, eps)
    U = np.multiply(u, np.sqrt(s_pinv))
    log_pdet = np.sum(np.log(s[s > eps]))

    return U, log_pdet


def logpdf(x, mean, prec_U, log_det_cov):
    """
    Log of the multivariate normal probability density function.

    Parameters -- these should be theano symbolic vars
    ----------
    x      : 2d matrix of observations    shape=(nbatch, obs_dim)
    mean   : 2d matrix of "means"         shape=(nbatch, obs_dim)
    prec_U : precision matrix             shape=(obs_dim, obs_dim)
    log_det_cov : log determinant of cov  (scalar)
    """
    dim = T.cast(x.shape[1], theano.config.floatX)
    log_2pi = 1.8378771
    dev = x - mean
    maha = T.sum(T.sqr(T.dot(dev, prec_U)), axis=1)
    out = -0.5 * (dim * log_2pi + log_det_cov + maha)
    return out
