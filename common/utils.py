import sys
sys.path.append('..')
import os
from common.import_library import *


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(np.power(grad, 2))
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split()

    id_to_word = {}
    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = [word_to_id[word] for word in words]
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(np.power(x, 2))) + eps)
    ny = y / (np.sqrt(np.sum(np.power(y, 2))) + eps)
    # return np.dot(nx, ny)
    return np.sum(nx * ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    
    print(f'\n[query] {query}')

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])

    cnt = 0
    for idx in (-1 * similarity).argsort():
        if id_to_word[idx] == query:
            continue
        print('%s: %s' % (id_to_word[idx], similarity[idx]))
        cnt += 1
        if cnt >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] + C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M
