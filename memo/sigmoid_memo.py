import sys
sys.path.append('..')
from common.import_library import *
# import numpy as np
# import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sig = sigmoid(np.arange(-5, 5, 0.1))
# plt.xlim(-5, 5)
plt.plot(sig, 'c-')
plt.show()
