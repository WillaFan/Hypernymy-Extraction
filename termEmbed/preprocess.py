#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib


def process_bless(filename):
    """Reads in BLESS.txt and returns word pairs."""
    hypernym_pairs = []
    cohypernym_pairs = []
    meronym_pairs = []
    random_pairs = []

    with open(filename, 'r') as file:
        for line in file:
            concept, _, relation, relatum = line.split()
            if relation == 'coord':
                cohypernym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'hyper':
                hypernym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'mero':
                meronym_pairs.append((relatum[:-2], concept[:-2]))
            elif relation == 'random-n':
                random_pairs.append((relatum[:-2], concept[:-2]))

    return hypernym_pairs, cohypernym_pairs, meronym_pairs, random_pairs

