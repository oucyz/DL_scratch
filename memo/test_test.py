import sys
import pytest
sys.path.append('..')
from common.import_library import *
from common.functions import softmax, cross_entropy_error, int_to_onehot
from common.utils import preprocess, create_co_matrix, cos_similarity

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

def test_int_to_onehot():
    labels1 = np.array([0, 1, 2, 3, 4])
    onehot_labels1 = int_to_onehot(labels1)
    # print(onehot_labels1)
    # print(np.eye(5))
    np.testing.assert_allclose(
        onehot_labels1,
        np.eye(5)
    )

    labels2 = np.array([3, 2, 5, 1, 9, 4])
    onehot_labels2 = int_to_onehot(labels2)
    # print(onehot_labels2)
    np.testing.assert_allclose(
        onehot_labels2,
        np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ])
    )

def test_cos_similarity():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    cs = cos_similarity(x, y)
    # assert(round(cs) == 1.0)
    np.testing.assert_allclose(cs, 1.0)

    x = np.array([[1], [2], [3]])
    y = np.array([[3], [2], [1]])
    cs = cos_similarity(x, y)
    np.testing.assert_allclose(cs, 0.714285714)

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    co = create_co_matrix(corpus, vocab_size)

    c0 = co[word_to_id['you']]
    c1 = co[word_to_id['i']]
    np.testing.assert_allclose(
        cos_similarity(c0, c1),
        0.7071067691154799
        )


# test_cross_entropy()
# test_softmax()
# test_int_to_onehot()
if __name__ == '__main__':
    pytest.main()
