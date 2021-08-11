import sys
sys.path.append('..')
from common.import_library import *


def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(-np.abs(x)))

def softmax(x):
    x -= x.max(axis=0, keepdims=True) # prevent overflow, x - C, C = x.max
    x = np.exp(x)
    x /= x.sum(axis=0, keepdims=True)
    return x

def cross_entropy_error(y_hat, y):
    delta = 1e-7
    batch_size = y_hat.shape[1]
    return -np.sum(y * np.log(y_hat + delta)) / batch_size
