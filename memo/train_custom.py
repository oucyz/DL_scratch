import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from models.two_layer_net import TwoLayerNet
from common.optimizer import SGD
from dataset import spiral


n_epochs = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
# (out_size, num_data) の形にする
x = x.transpose()
t = t.transpose()
print('x =', x.shape)
print('t =', t.shape)

model = TwoLayerNet(2, hidden_size, 3)
optimizer = SGD(lr=learning_rate)

data_size = x.shape[1]
n_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(1, n_epochs+1):
    idx = np.random.permutation(data_size)
    x = x[:, idx]
    t = t[:, idx]

    for iters in range(n_iters):
        batch_x = x[:, iters*batch_size:(iters+1)*batch_size]
        batch_t = t[:, iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if epoch % 10 == 0:
            avg_loss = total_loss / loss_count
            print('Epoch %d, iter %d / %d, Loss %.3f'\
                % (epoch, iters + 1, n_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss = 0
            loss_count = 0

plt.figure()
plt.plot(loss_list, 'r-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
