import sys
sys.path.append('..')
from common.import_library import *
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        self.x = x
        W, = self.params
        out = np.dot(W, x)
        return out
    
    def backward(self, dout):
        W, = self.params
        dW = np.dot(dout, self.x.T)
        dx = np.dot(W.T, dout)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out*(1 - self.out)
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        self.x = x
        W, b = self.params
        out = np.dot(W, x) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dW = np.dot(dout, self.x.T)
        db = np.sum(dout, axis=1, keepdims=True)
        dx = np.dot(W.T, dout)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None # predict
        self.t = None # label
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.y.shape[1]
        dx = (self.y - self.t) * dout / batch_size
        return dx
