import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x.shape =', x.shape) # (300, 2)
print('t.shape =', t.shape) # (300, 2)

xlabels = t.argmax(axis=1)
x1 = x[xlabels == 0]
x2 = x[xlabels == 1]
x3 = x[xlabels == 2]

plt.figure()
plt.plot(x1[:, 0], x1[:, 1], 'c.')
plt.plot(x2[:, 0], x2[:, 1], 'r.')
plt.plot(x3[:, 0], x3[:, 1], 'g.')
plt.show()
