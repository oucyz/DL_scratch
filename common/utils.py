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
