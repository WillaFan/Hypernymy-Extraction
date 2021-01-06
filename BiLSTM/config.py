import numpy as np
import random
import torch


def apply_random_seed():
    seed=1206
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class config:
    use_f1 = True
    use_char_embedding = True
    use_modified_LSTMCell = True

    train_file = './data/train.txt'
    dev_file = './data/dev.txt'
    test_file = './data/test.txt'
    tag_file = './data/tags.txt'
    char_embedding_file = './data/char_embeddings.txt'
    word_embedding_file = './data/word_embeddings.txt'
    model_file = './result_model.pt'

    word_embedding_dim = 50
    char_embedding_dim = 50
    char_lstm_output_dim = 50
    batch_size = 10
    hidden_dim = 50
    nepoch = 10
    dropout = 0.5

    nwords = 0
    nchars = 0
    ntags = 0
