import sys
import pytest
sys.path.append('..')
from common.import_library import *
from common.functions import softmax, cross_entropy_error


batch_size = 1 # batch size
out_size = 2

def test_cross_entropy():
    labels = np.random.randint(0, out_size, (batch_size))
    # print(labels)
    onehot_labels = np.zeros((out_size, labels.shape[0]))
    # print(onehot_labels)
    for i in range(len(labels)):
        onehot_labels[:, i] = np.eye(out_size)[labels[i]]
    # print(onehot_labels)

    y_hat = np.abs(np.random.random_sample((out_size, batch_size)))
    # print(y_hat)
    # print(cross_entropy_error(y_hat, onehot_labels))

def test_softmax():
    x1 = np.array([[0], [1]])
    out1 = softmax(x1)
    # print('x1 =')
    # print(x1)
    # print('out1 =')
    # print(out1)
    assert(all(np.round(softmax(x1), 7) == np.array([[0.2689414], [0.7310586]])))
    x2 = np.array([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]])
    out2 = softmax(x2)
    # print(x2, out2)
    np.testing.assert_allclose(
        softmax(x2),
        np.array([[0.00235563, 0.00235563, 0.00235563], [0.04731416, 0.04731416, 0.04731416], [0.95033021, 0.95033021, 0.95033021]]),
        rtol=1e-5)

# test_softmax()
if __name__ == '__main__':
    pytest.main()
