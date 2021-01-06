import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from config import config, apply_random_seed
from preprocess import DataReader, gen_embedding_from_file, read_tag_vocab
import torch
import random

_config = config()
apply_random_seed()


def evaluate(golden_list, predict_list):

    if all(not g for g in golden_list) and all(not p for p in predict_list):
        return 1

    fp, fn, tp = 0, 0, 0
    Blist, Ilist = ['B-TAR', 'B-HYP'], ['I-TAR', 'I-HYP']

    for j in range(len(golden_list)):
        for i in range(len(golden_list[j])):
            if predict_list[j][i] in Blist and golden_list[j][i] != predict_list[j][i]:
                fp += 1
            if golden_list[j][i] in Blist:
                if golden_list[j][i] != predict_list[j][i]:
                    fn += 1
                else:
                    if i != len(golden_list[j]) -1:
                        for n in range(i+1, len(golden_list[j])):
                            if golden_list[j][n] not in Ilist and predict_list[j][n] not in Ilist:
                                tp+=1
                                break
                            elif golden_list[j][n] != predict_list[j][n]:
                                fn += 1
                                fp += 1
                                break
                            elif n==len(golden_list[j])-1 and golden_list[j][n] == predict_list[j][n]:
                                tp += 1
                    else:
                        tp += 1

    try:
        f1 = (2*tp)/(2*tp + fn + fp)
    except:
        f1 = 0
    return f1


def test():
    tag_dict = read_tag_vocab(_config.tag_file)
    reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
    word_embedding, word_dict = gen_embedding_from_file(_config.word_embedding_file, _config.word_embedding_dim)
    char_embedding, char_dict = gen_embedding_from_file(_config.char_embedding_file, _config.char_embedding_dim)

    # load the pretrained model
    model = torch.load('./result_model.pt')
    test = DataReader(_config, _config.test_file, word_dict, char_dict, tag_dict, _config.batch_size)
    pred_dev_ins, golden_dev_ins = [], []
    for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in test:
        pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
        pred_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
        golden_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]

    return pred_dev_ins, golden_dev_ins


def writer(filepath,preds):
    with open(filepath, "w", encoding="utf-8") as f_out, open("./data/test.txt", "r", encoding="utf-8") as f:
        sens = []; t = []
        for line in f:
            if line!='\n':
                t.append(line)
            else:
                sens.append(t)
                t=[]

#         pairs=[]
        for sen,pred in zip(sens,preds):
            if len(sen) == len(pred):
                _ = []
                for i in range(len(sen)):
                    if pred[i] != 'O':
                        _.append(sen[i].split()[0]+" "+pred[i])
#                 pairs.append(_)
                f_out.write(', '.join(_)+'\n')
    f.close()
    f_out.close()