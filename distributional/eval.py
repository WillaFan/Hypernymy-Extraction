#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import average_precision_score


def ap(y_true, y_score):
    """
    Average precision score.
    """
    return average_precision_score(y_true, y_score)


def ap_at_k(y_true, y_score, k):
    """
    Computes AP@k, or AP of the model's top K predictions.
    """
    argsort = np.argsort(y_score)
    score_srt = np.array(y_score)[argsort[-k:]]
    label_srt = np.array(y_true)[argsort[-k:]]
    
    return average_precision_score(label_srt, score_srt)

