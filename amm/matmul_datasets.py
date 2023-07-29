import sys
import os
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, '..', 'dataset', 'cifar10-softmax')

class MatmulTask(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, W_train, W_test=None,
                 name=None, info=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.W_train = W_train
        self.W_test = W_test if W_test is not None else W_train
        self.name = name
        self.info = info if info is not None else {}

        self.train_mats = (self.X_train, self.Y_train, self.W_train)
        self.test_mats = (self.X_test, self.Y_test, self.W_test)
        self.initial_hashes = self._hashes()

    def __str__(self):
        train_str = '{} @ {} = {}'.format(
            self.X_train.shape, self.W_train.shape, self.Y_train.shape)
        test_str = '{} @ {} = {}'.format(
            self.X_test.shape, self.W_test.shape, self.Y_test.shape)
        s = "train:\t{}\ntest:\t{}".format(train_str, test_str)
        if self.name:
            s = "---- {}\n{}".format(self.name, s)
        return s

    def validate_shapes(self):
        for (X, Y, W) in [self.train_mats, self.test_mats]:
            N, D = X.shape
            D2, M = W.shape
            assert D == D2
            assert (N, M) == Y.shape

    def validate_hashes(self):
        assert self._hashes() == self.initial_hashes

    def validate(self, verbose=1, mse_thresh=1e-7, train=True, test=True):
        self.validate_shapes()
        self.validate_hashes()

        which_mats = []
        if train:
            which_mats.append(self.train_mats)
        if test:
            which_mats.append(self.test_mats)

        for (X, Y, W) in which_mats:
            Y_hat = X @ W
            diffs = Y - Y_hat
            mse = np.mean(diffs * diffs) / np.var(Y)
            if verbose > 0:
                print("mse: ", mse)
            assert mse < mse_thresh

    def _hashes(self):
        return {
            'X_train': self.X_train.std(),
            'Y_train': self.Y_train.std(),
            'W_train': self.W_train.std(),
            'X_test': self.X_test.std(),
            'Y_test': self.Y_test.std(),
            'W_test': self.W_test.std()
        }



def load_mat(fname):
    fpath = os.path.join(CIFAR10_DIR, fname)
    return np.load(fpath)

def load_cifar10_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar10_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar10_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar10_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar10_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar10_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar10_softmax_b.npy'
    LABELS_TRAIN_PATH = 'cifar10_labels_train.npy'
    LABELS_TEST_PATH = 'cifar10_labels_test.npy'


    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel() #lbls:(50000,1) 2d; .ravel() make it 1D: (50000)
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # X_Train.dot(W)+b=Y_train

    Y_train -= b
    Y_test -= b

    ##%%
    '''
    # TODO rm all this after debug
    logits_test = Y_test + b

    print("logits_test.shape", logits_test.shape)
    print("lbls_test.shape", lbls_test.shape)
    lbls_hat_test = np.argmax(Y_test, axis=1)
    print("lbls_hat_test.shape", lbls_hat_test.shape)
    acc = np.mean(lbls_hat_test.ravel() == lbls_test.ravel())
    print("Y_test: ", Y_test[:10])
    print("Y_train head: ", Y_train[:10])
    print("Y_train tail: ", Y_train[-10:])
    print("b:\n", b)
    # print("lbls hat test:")
    # print(lbls_hat_test[:100])
    # print("lbls test:")
    # print(lbls_test[:100])
    print("lbls train:")
    print(lbls_train[:100])
    print("acc: ", acc)
    '''

    info = {'problem': 'softmax', 'biases': b,
            'lbls_train': lbls_train, 'lbls_test': lbls_test}

    return MatmulTask(X_train, Y_train, X_test, Y_test, W,
                       name='CIFAR-10 Softmax', info=info)