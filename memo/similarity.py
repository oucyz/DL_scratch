import sys
sys.path.append('..')
from common.import_library import *
from common.utils import preprocess, create_co_matrix, cos_similarity, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
co = create_co_matrix(corpus, vocab_size)

c0 = co[word_to_id['you']]
c1 = co[word_to_id['i']]
print("similarity 'You' and 'I' =", cos_similarity(c0, c1))


for i in range(vocab_size):
    query = id_to_word[i]
    most_similar(query, word_to_id, id_to_word, co)
