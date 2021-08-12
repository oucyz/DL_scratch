import sys
sys.path.append('..')
import time
from common.import_library import *
from common.utils import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, n_epochs=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = x.shape[1]
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(n_epochs):
            idx = np.random.permutation(data_size)
            x = x[:, idx]
            t = t[:, idx]

            for iters in range(max_iters):
                batch_x = x[:, iters*batch_size:(iters+1)*batch_size]
                batch_t = t[:, iters*batch_size:(iters+1)*batch_size]

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = model.params, model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval == 0):
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('Epoch %d, iter %d / %d, time %d[s], Loss %.3f'\
                        % (self.current_epoch+1, iters+1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
                
        self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, 'r-', label='train')
        plt.xlabel(f'iterations (x{str(self.eval_interval)})')
        plt.ylabel('loss')
        plt.show()
