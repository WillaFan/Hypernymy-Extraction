#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Hyperparameter settings (model)
nb_maps = 24
dropout = 0.5
normalize_e = False
normalize_p = False

# Hyperparameter settings (optimizer)
learning_rate = 2e-4
beta1 = 0.9
beta2 = 0.9
weight_decay = 0

# Hyperparameter settings (training algorithm)
nb_neg_samples = 10
subsample = True
max_epochs = 200
patience = 200
batch_size = 16
clip = 1e-4

hparams={'nb_maps':nb_maps,'dropout':dropout,'normalize_e':normalize_e,'normalize_p':normalize_p,
         'learning_rate':learning_rate,'beta1':beta1,'beta2':beta2,'weight_decay':weight_decay,
         'nb_neg_samples':nb_neg_samples,'subsample':subsample,'max_epochs':max_epochs,'patience':patience,'batch_size':batch_size,'clip':clip}

