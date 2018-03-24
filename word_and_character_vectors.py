from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np
import os

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1

#these are the lines in the glove and fasttext files
GLOVE_VOCAB_SIZE=1917494
GLOVE_FILENAME='glove.42B.300d.txt'
GLOVE_DIMENSION=300
FASTTEXT_VOCAB_SIZE=2000000
FASTTEXT_FILENAME='crawl-300d-2M.vec'
FASTTEXT_DIMENSION=300

#get glove vectors. need to pass just the data path location
#adds glove.42B.300d to path
def get_glove(data_file_path):
    path=os.path.join(data_file_path,GLOVE_FILENAME)
    return get_word_embeddings(path,GLOVE_VOCAB_SIZE,GLOVE_DIMENSION)

#gets fasttext vectors. need to pass just the data path location
#adds glove.42B.300d to path
def get_fasttext(data_file_path):
    path = os.path.join(data_file_path, FASTTEXT_FILENAME)
    return get_word_embeddings(path, FASTTEXT_VOCAB_SIZE,FASTTEXT_DIMENSION)

def get_word_embeddings(datafile,vocab_size,dimension):
    """Reads from the data file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      datafile: path to data file
      vocab_size: size of vocabulary
      dimension: vector dimension

    Returns:
      emb_matrix: Numpy array shape (vocab_size, dimension) containing vector embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """
    print ("Loading vectors from file: %s" % datafile)
    dim = dimension

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(datafile, 'r', encoding="utf8") as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

