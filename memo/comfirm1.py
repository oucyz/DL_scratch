import sys
sys.path.append('..')
from common.import_library import *
from common.layers import Affine, Sigmoid


m = 10
n0 = 2
n1 = 3
x = np.random.randn(2, m)
W = np.random.randn(n1, n0)
b = np.random.randn(n1, 1)
layers = [Affine(W, b),
          Sigmoid()]

for layer in layers:
    x = layer.forward(x)
print(x.shape)

dx = np.random.randn(n1, m)
for layer in reversed(layers):
    dx = layer.backward(dx)
print(dx.shape)
