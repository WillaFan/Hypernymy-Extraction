#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import logging
import os
import gzip
import scipy.sparse as sp


# Helper functions
def __try_three_columns(string):
    fields=string.split('\t')
    if len(fields)>3:
        fields=fields[:3]
    if len(fields)==3:
        return fields[0],fields[1],float(fields[2])
    if len(fields==2):
        return fields[0],fields[1],1.0
    else:
        raise ValueError('Invalid number of fields {}'.format(len(fields)))

def __load_sparse_matrix(filename,same_vocab):
    """ 
    Actual workhorse of the model.
    """
    objects=['<OOV>']
    rowvocab={'<OOV>':0}
    if same_vocab:
        colvocab=rowvocab
    else:
        colvocab={}
    _is=[]
    _js=[]
    _vs=[]
    
    if filename.endswith('.gz'):
        f=gzip.open(filename,'rb')
    else:
        f=open(filename,'rb')
        
    for line in f:
        line=line.decode('utf-8')
        target,context,weight=__try_three_columns(line)
        if target not in rowvocab:
            rowvocab[target]=len(rowvocab)
            objects.append(target)
        if context not in colvocab:
            colvocab[context]=len(colvocab)
            if same_vocab:
                objects.append(context)
        _is.append(rowvocab[target])
        _js.append(colvocab[context])
        _vs.append(weight)  # weight: w(x,y)
        
    f.close()
    
    _shape=(len(rowvocab),len(colvocab))
    spmatrix=sp.csr_matrix((_vs,(_is,_js)),shape=_shape,dtype=np.float64)
    return spmatrix,objects,rowvocab,colvocab
        
def read_sparse_matrix(filename,allow_binary_cache=False,same_vocab=False):
    cache_filename=filename+'.pkl'
    cache_exists=os.path.exists(cache_filename)
    cache_fresh=cache_exists and os.path.getmtime(filename)<=os.path.getmtime(cache_filename)
    if allow_binary_cache and cache_fresh:
        logging.debug('Using space cache {}'.format(cache_filename))
        with open(cache_filename+'pkl','rb') as pklf:
            return pickle.load(pklf)
    else:
        result=__load_sparse_matrix(filename,same_vocab=same_vocab)
        if allow_binary_cache:
            logging.warning('Dumping the binary cache {}.pkl'.format(filename))
            with open(filename+'pkl','wb') as pklf:
                pickle.dump(result,pklf)
        return result

