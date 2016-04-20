import numpy as np
import numpy.random as npr
import scipy.misc as scipy_misc
import os
import sys
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt
import cPickle
import gzip
import theano

from lib.data_utils import shuffle
from lib.rng import py_rng, np_rng, t_rng, cu_rng, set_seed

def row_shuffle(X):
    """
    Return a copy of X with shuffled rows.
    """
    shuf_idx = np.arange(X.shape[0])
    npr.shuffle(shuf_idx)
    X_shuf = X[shuf_idx]
    return X_shuf

def mnist(data_dir):
    fd = open("{}/train-images.idx3-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open("{}/train-labels.idx1-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open("{}/t10k-images.idx3-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open("{}/t10k-labels.idx1-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set(data_dir):
    trX, teX, trY, teY = mnist(data_dir)

    trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY

def load_binarized_mnist(data_path='./'):
    #binarized_mnist_test.amat  binarized_mnist_train.amat  binarized_mnist_valid.amat
    print 'loading binary MNIST, sampled version (de Larochelle)'
    train_x = np.loadtxt(data_path + 'binarized_mnist_train.amat').astype('float32')
    valid_x = np.loadtxt(data_path + 'binarized_mnist_valid.amat').astype('float32')
    test_x = np.loadtxt(data_path + 'binarized_mnist_test.amat').astype('float32')
    # shuffle dataset
    train_x = row_shuffle(train_x)
    valid_x = row_shuffle(valid_x)
    test_x = row_shuffle(test_x)
    return train_x, valid_x, test_x

def load_svhn(tr_file, te_file, ex_file=None, ex_count=None):
    """
    Loads the full SVHN train/test sets and an additional number of randomly
    selected examples from the "extra set".
    """
    import gc
    import scipy.io as io
    # load the training set as a numpy arrays
    data_dict = io.loadmat(tr_file)
    Xtr = data_dict['X'].astype(theano.config.floatX)
    Ytr = data_dict['y'].astype(np.int32)
    Xtr_vec = np.zeros((Xtr.shape[3], 32*32*3), dtype=theano.config.floatX)
    for i in range(Xtr.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xtr_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xtr[:,:,c,i].reshape((32*32,))
    Xtr = Xtr_vec
    del data_dict
    gc.collect()
    # load the test set as numpy arrays
    data_dict = io.loadmat(te_file)
    Xte = data_dict['X'].astype(theano.config.floatX)
    Yte = data_dict['y'].astype(np.int32)
    Xte_vec = np.zeros((Xte.shape[3], 32*32*3), dtype=theano.config.floatX)
    for i in range(Xte.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xte_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xte[:,:,c,i].reshape((32*32,))
    Xte = Xte_vec
    del data_dict
    gc.collect()
    if ex_file is None:
        Xex = None
    else:
        # load the extra digit examples and only keep a random subset
        data_dict = io.loadmat(ex_file)
        ex_full_size = data_dict['X'].shape[3]
        idx = np.arange(ex_full_size)
        npr.shuffle(idx)
        idx = idx[:ex_count]
        Xex = data_dict['X'].take(idx, axis=3).astype(theano.config.floatX)
        Xex_vec = np.zeros((Xex.shape[3], 32*32*3), dtype=theano.config.floatX)
        for i in range(Xex.shape[3]):
            c_pix = 32*32
            for c in range(3):
                Xex_vec[i,c*c_pix:((c+1)*c_pix)] = \
                        Xex[:,:,c,i].reshape((32*32,))
        Xex = Xex_vec
        del data_dict
        gc.collect()

    print("np.max(Xtr): {0:.4f}, np.min(Xtr): {1:.4f}".format(np.max(Xtr), np.min(Xtr)))

    # package data up for easy returnage
    data_dict = {'Xtr': Xtr, 'Ytr': Ytr, \
                 'Xte': Xte, 'Yte': Yte, \
                 'Xex': Xex}
    return data_dict

def load_svhn_ss(tr_file, te_file, ex_file=None, ex_count=None):
    """
    Loads the full SVHN train/test sets and an additional number of randomly
    selected examples from the "extra set".
    """
    import gc
    import scipy.io as io
    # load the training set as a numpy arrays
    data_dict = io.loadmat(tr_file)
    Xtr = data_dict['X'].astype(theano.config.floatX)
    Ytr = data_dict['y'].astype(np.int32)
    Xtr_vec = np.zeros((Xtr.shape[3], 32*32*3), dtype=theano.config.floatX)
    for i in range(Xtr.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xtr_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xtr[:,:,c,i].reshape((32*32,))
    Xtr = Xtr_vec
    del data_dict
    gc.collect()
    # load the test set as numpy arrays
    data_dict = io.loadmat(te_file)
    Xte = data_dict['X'].astype(theano.config.floatX)
    Yte = data_dict['y'].astype(np.int32)
    Xte_vec = np.zeros((Xte.shape[3], 32*32*3), dtype=theano.config.floatX)
    for i in range(Xte.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xte_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xte[:,:,c,i].reshape((32*32,))
    Xte = Xte_vec
    del data_dict
    gc.collect()
    if ex_file is None:
        Xex = None
    else:
        # load the extra digit examples and only keep a random subset
        data_dict = io.loadmat(ex_file)
        ex_full_size = data_dict['X'].shape[3]
        idx = np.arange(ex_full_size)
        npr.shuffle(idx)
        idx = idx[:ex_count]
        Xex = data_dict['X'].take(idx, axis=3).astype(theano.config.floatX)
        Xex_vec = np.zeros((Xex.shape[3], 32*32*3), dtype=theano.config.floatX)
        for i in range(Xex.shape[3]):
            c_pix = 32*32
            for c in range(3):
                Xex_vec[i,c*c_pix:((c+1)*c_pix)] = \
                        Xex[:,:,c,i].reshape((32*32,))
        Xex = Xex_vec
        del data_dict
        gc.collect()

    print("np.max(Xtr): {0:.4f}, np.min(Xtr): {1:.4f}".format(np.max(Xtr), np.min(Xtr)))

    # package data up for easy returnage
    data_dict = {'Xtr': Xtr, 'Ytr': Ytr, \
                 'Xte': Xte, 'Yte': Yte, \
                 'Xex': Xex}
    return data_dict

def load_udm(dataset, to_01=False):
    """Loads the UdM train/validate/test split of MNIST."""
    # Download the MNIST dataset if it is not present
    print("Loading real-valued MNIST data...")
    data_dir, data_file = os.path.split(dataset)
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # old, ugly code
    train_set = [v for v in train_set]
    valid_set = [v for v in valid_set]
    test_set = [v for v in test_set]
    train_set[0] = np.asarray(train_set[0])
    valid_set[0] = np.asarray(valid_set[0])
    test_set[0] = np.asarray(test_set[0])
    if to_01:
        # training set...
        train_set[0] = train_set[0] - np.min(train_set[0])
        train_set[0] = train_set[0] / np.max(train_set[0])
        # validation set...
        valid_set[0] = valid_set[0] - np.min(valid_set[0])
        valid_set[0] = valid_set[0] / np.max(valid_set[0])
        # test set...
        test_set[0] = test_set[0] - np.min(test_set[0])
        test_set[0] = test_set[0] / np.max(test_set[0])

    # old, ugly code
    test_set_x, test_set_y = test_set[0], test_set[1]
    valid_set_x, valid_set_y = valid_set[0], valid_set[1]
    train_set_x, train_set_y = train_set[0], train_set[1]
    # wtf, ugly code
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_udm_ss(dataset, sup_count, im_dim=28):
    """
    Load semi-supervised version of the standard UdM MNIST data.

    For this, the training data is split into labeled and unlabeled portions.
    The number of labeled examples is 'sup_count', and an equal number of
    labeled examples will be selected for each class. The remaining (50000 -
    sup_count) examples are provided as unlabeled training data. The validate
    and test sets are left unchanged.
    """

    udm_data = load_udm(dataset)
    Xtr = udm_data[0][0]
    Ytr = udm_data[0][1][:,np.newaxis]

    all_count = Xtr.shape[0]
    pc_count = int(np.ceil(sup_count / 10.0))
    sup_count = int(10 * pc_count)
    unsup_count = all_count - sup_count

    Xtr_su = []
    Ytr_su = []
    Xtr_un = []
    Ytr_un = []

    # Sample supervised and unsupervised subsets of each class' observations
    for c_label in np.unique(Ytr):
        c_idx = [i for i in range(all_count) if (Ytr[i] == c_label)]
        np_rng.shuffle(c_idx)
        Xtr_su.append(Xtr[c_idx[0:pc_count],:])
        Ytr_su.append(Ytr[c_idx[0:pc_count],:])
        Xtr_un.append(Xtr[c_idx[pc_count:],:])
        Ytr_un.append(Ytr[c_idx[pc_count:],:])

    # Stack per-class supervised/unsupervised splits into matrices
    Xtr_su = np.vstack(Xtr_su)
    Ytr_su = np.vstack(Ytr_su) # these labels will be used in training...
    Xtr_un = np.vstack(Xtr_un)
    Ytr_un = np.vstack(Ytr_un) # these labels won't be used in training...

    # shuffle the rows so that observations are not grouped by class
    shuf_idx = np_rng.permutation(Xtr_su.shape[0])
    Xtr_su = Xtr_su[shuf_idx,:]
    Ytr_su = Ytr_su[shuf_idx].ravel()
    shuf_idx = np_rng.permutation(Xtr_un.shape[0])
    Xtr_un = Xtr_un[shuf_idx,:]
    Ytr_un = Ytr_un[shuf_idx].ravel()

    # grab the validation and test data
    Xva, Yva = udm_data[1][0], udm_data[1][1]
    Xte, Yte = udm_data[2][0], udm_data[2][1]

    # guarantee the type of the images
    max_x = max([np.max(Xtr_su), np.max(Xtr_un), np.max(Xva), np.max(Xte)])
    Xtr_su = Xtr_su.astype(theano.config.floatX) / max_x
    Xtr_un = Xtr_un.astype(theano.config.floatX) / max_x
    Xva = Xva.astype(theano.config.floatX) / max_x
    Xte = Xte.astype(theano.config.floatX) / max_x

    # resize images if desired
    if not (im_dim == 28):
        new_shape = (im_dim, im_dim)
        old_shape = (28,28)
        print("resizing images from {} to {}".format(old_shape, new_shape))

        print("  -- resizing Xtr_su...")
        Xtr_su_new = np.zeros((Xtr_su.shape[0], im_dim*im_dim),
                              dtype=theano.config.floatX)
        for i in range(Xtr_su.shape[0]):
            x_old = Xtr_su[i,:].reshape(old_shape)
            x_new = scipy_misc.imresize(x_old, size=new_shape, mode='F')
            Xtr_su_new[i,:] = x_new.ravel()
        Xtr_su = Xtr_su_new

        print("  -- resizing Xtr_un...")
        Xtr_un_new = np.zeros((Xtr_un.shape[0], im_dim*im_dim),
                              dtype=theano.config.floatX)
        for i in range(Xtr_un.shape[0]):
            x_old = Xtr_un[i,:].reshape(old_shape)
            x_new = scipy_misc.imresize(x_old, size=new_shape, mode='F')
            Xtr_un_new[i,:] = x_new.ravel()
        Xtr_un = Xtr_un_new

        print("  -- resizing Xva...")
        Xva_new = np.zeros((Xva.shape[0], im_dim*im_dim),
                              dtype=theano.config.floatX)
        for i in range(Xva.shape[0]):
            x_old = Xva[i,:].reshape(old_shape)
            x_new = scipy_misc.imresize(x_old, size=new_shape, mode='F')
            Xva_new[i,:] = x_new.ravel()
        Xva = Xva_new

        print("  -- resizing Xte...")
        Xte_new = np.zeros((Xte.shape[0], im_dim*im_dim),
                              dtype=theano.config.floatX)
        for i in range(Xte.shape[0]):
            x_old = Xte[i,:].reshape(old_shape)
            x_new = scipy_misc.imresize(x_old, size=new_shape, mode='F')
            Xte_new[i,:] = x_new.ravel()
        Xte = Xte_new

    # package the data for the user
    data_dict = {'Xtr_su': Xtr_su, 'Ytr_su': Ytr_su,
                 'Xtr_un': Xtr_un, 'Ytr_un': Ytr_un,
                 'Xva': Xva, 'Yva': Yva,
                 'Xte': Xte, 'Yte': Yte}
    return data_dict



def one_hot(x, n):
    '''
    convert index representation to one-hot representation
    '''
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def _load_batch_cifar10(batch_dir, batch_name, dtype='float32'):
    '''
    load a batch in the CIFAR-10 format
    '''
    path = os.path.join(batch_dir, batch_name)
    batch = np.load(path)
    data = batch['data'] / 255.0             # scale between [0, 1]
    labels = one_hot(batch['labels'], n=10)  # convert labels to one-hot
    return data.astype(dtype), labels.astype(dtype)


def load_cifar10(data_dir, va_split=5000, dtype='float32', grayscale=False):
    dir_cifar10 = os.path.join(data_dir, 'cifar-10-batches-py')
    #class_names_cifar10 = np.load(os.path.join(dir_cifar10, 'batches.meta'))

    # train
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10(dir_cifar10, 'data_batch_{}'.format(k + 1),
                                   dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10(dir_cifar10, 'test_batch', dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    if (va_split is not None) and (va_split > 0):
        x_src = x_train.copy()
        t_src = t_train.copy()
        x_train = x_src[va_split:]
        t_train = t_src[va_split:]
        x_test = x_src[:va_split]
        t_test = x_src[:va_split]

    return x_train, t_train, x_test, t_test
