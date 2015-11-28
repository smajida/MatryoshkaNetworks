import numpy as np
import numpy.random as npr
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt
import cPickle
import theano

from lib.data_utils import shuffle

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

def load_svhn(tr_file, te_file, ex_file=None, ex_count=None):
    """
    Loads the full SVHN train/test sets and an additional number of randomly
    selected examples from the "extra set".
    """
    # load the training set as a numpy arrays
    pickle_file = open(tr_file)
    data_dict = cPickle.load(pickle_file)
    Xtr = data_dict['X'].astype(theano.config.floatX)
    Ytr = data_dict['y'].astype(np.int32) + 1
    Xtr_vec = np.zeros((Xtr.shape[3], 32*32*3)).astype(theano.config.floatX)
    for i in range(Xtr.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xtr_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xtr[:,:,c,i].reshape((32*32,))
    Xtr = Xtr_vec
    pickle_file.close()
    # load the test set as numpy arrays
    pickle_file = open(te_file)
    data_dict = cPickle.load(pickle_file)
    Xte = data_dict['X'].astype(theano.config.floatX)
    Yte = data_dict['y'].astype(np.int32) + 1
    Xte_vec = np.zeros((Xte.shape[3], 32*32*3)).astype(theano.config.floatX)
    for i in range(Xte.shape[3]):
        c_pix = 32*32
        for c in range(3):
            Xte_vec[i,c*c_pix:((c+1)*c_pix)] = \
                    Xte[:,:,c,i].reshape((32*32,))
    Xte = Xte_vec
    pickle_file.close()
    if ex_file is None:
        Xex = None
    else:
        # load the extra digit examples and only keep a random subset
        pickle_file = open(ex_file)
        data_dict = cPickle.load(pickle_file)
        ex_full_size = data_dict['X'].shape[3]
        idx = np.arange(ex_full_size)
        npr.shuffle(idx)
        idx = idx[:ex_count]
        Xex = data_dict['X'].take(idx, axis=3).astype(theano.config.floatX)
        Xex_vec = np.zeros((Xex.shape[3], 32*32*3)).astype(theano.config.floatX)
        for i in range(Xex.shape[3]):
            c_pix = 32*32
            for c in range(3):
                Xex_vec[i,c*c_pix:((c+1)*c_pix)] = \
                        Xex[:,:,c,i].reshape((32*32,))
        Xex = Xex_vec
        pickle_file.close()

    print("np.max(Xtr): {0:.4f}, np.min(Xtr): {1:.4f}".format(np.max(Xtr), np.min(Xtr)))

    # package data up for easy returnage
    data_dict = {'Xtr': Xtr, 'Ytr': Ytr, \
                 'Xte': Xte, 'Yte': Yte, \
                 'Xex': Xex}
    return data_dict