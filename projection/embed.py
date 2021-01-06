#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec
import numpy as np
import codecs


def train_embed(read_corpus_path, save_embed_path, **kwargs):
    """
    Args specific to word embedings must be the same as Word2Vec.
    """
    
    preprocessed_corpus = []
    for line in open(read_corpus_path, "r", encoding="utf-8"):
        preprocessed_corpus.append(line)
    tokens = [line.split() for line in preprocessed_corpus]
    
    params_space = {
        'size': None,
        'window': None,
        'min_count': None,
        'workers': None,
        'sg': None,
        'negative': None,
        'sample': None,
        'epochs': None
    }
    for param, value in kwargs.items():
        if param in params_space.keys():
            params_space[param]=value
    
    print("Training embeddings...")
    word2vec_model = Word2Vec(tokens, size=params_space['size'], window=params_space['window'], min_count=params_space['min_count'],
                              workers=params_space['workers'], sg=params_space['sg'], negative=params_space['negative'], sample=params_space['sample'])  # epochs=10
    
    # Save the word embeddings
    print("Saving the model...")
    word2vec_model.wv.save_word2vec_format(save_embed_path, binary=True)  # binary format
    print("Finished.")
    
def load_embed(embed_path):
    
    return word2vec.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    
def write_embed(load_embed_path, write_embed_path):
    embed_model = load_embed(load_embed_path)
    with open(write_embed_path, "w", encoding="utf-8") as f_embed:    
        for word in list(embed_model.wv.vocab):
            wvec=[word]
            for ndim in embed_model.wv[word]:
                wvec.append(str(ndim))
            f_embed.write(" ".join(wvec)+"\n")
    f_embed.close()
    
def get_embeddings(path, dtype=np.float32):
    """Get word embeddings from text file. 
    Args:
    - path
    - dtype: dtype of matrix
    Returns:
    - vocab (list of words) 
    - dict that maps words to vectors
    """
    # Get vector size
    with codecs.open(path, "r", "utf-8") as f:
        elems = f.readline().strip().split()
        if len(elems) == 2:
            header = True
            dim = int(elems[1])
        else:
            header = False
            dim = len(elems)-1
    words = []
    word2vec = {}
    with codecs.open(path, "r", "utf-8") as f:
        line_count = 0
        if header:
            f.readline()
            line_count = 1
        for line in f:
            line_count += 1
            elems = line.strip().split()
            if len(elems) == dim + 1:
                word = elems[0]
                try:
                    vec = np.asarray([float(i) for i in elems[1:]], dtype=dtype)
                    words.append(word)
                    word2vec[word] = vec
                except ValueError as e:
                    print("ValueError: Skipping line {}".format(line_count))
            else:
                msg = "Error: Skipping line {}. ".format(line_count)
                msg += "Expected {} elements, found {}.".format(dim+1, len(elems))
                print(msg)
    return words, word2vec

def make_embedding_matrix(word2vec, words, seed=0):
    """ Given a mapping of words to vectors and a list of words, make
    a matrix containing the vector of each word. Assign a random
    vector to words that don't have one.
    Args:
    - word2vec: dict that maps words to vectors
    - words: list of words
    - seed: seed for numpy's RNG
    Returns:
    - matrix: containing row vector for each word in same order as the
      list of words
    """
    np.random.seed(seed)
    for word in words:
        if word in word2vec:
            dim = word2vec[word].shape[0]
            dtype = word2vec[word].dtype
            break
    matrix = np.zeros((len(words), dim), dtype=dtype)
    for (i,word) in enumerate(words):
        if word in word2vec:
            matrix[i] = word2vec[word]
        else:
            matrix[i] = np.random.uniform(low=-0.5, high=0.5) / dim
    return matrix

def normalize_numpy_matrix(x):
    """ Make rows in a 2-D numpy array unit-length. """
    return x / np.linalg.norm(x, ord=2, axis=1).reshape(-1,1)

def make_pairs(queries, hyps, query2id, hyp2id):
    """ Given a list of queries, a list of lists of gold hypernyms, a
    dict that maps from queries to IDs and a dict that maps from
    hypernyms to IDs, make a list of (query ID, hypernym ID)
    pairs."""
    pairs = []
    # print("-"*5,len(queries))
    # print("-"*5,len(hyps))
    for i in range(len(queries)):
        q = queries[i]
        q_id = query2id[q]
        if i <= len(hyps):
            h = hyps[i]
            h_id = hyp2id[h]
            pairs.append([q_id, h_id])
    pairs = np.array(pairs, dtype=np.int)
    return pairs
