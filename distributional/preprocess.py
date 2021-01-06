#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
sys.path.append("..")
import pickle

"""
    Preprocessing for distributional models.
"""

def read_vocab(file_path):
    """Returns the lookup in between vocab:
        entityID and PreferredName, AllNames.
    """
    
    entityID2entityPreferredName={}
    entityID2AllNames={}
    with open(file_path,"r") as f_in:
        for l_id,line in enumerate(f_in):
            if l_id==0:  # skip header
                continue
            line=line.strip()
            if line:
                segs=line.split("\t")
                entityID=segs[0]
                entityPreferredName=segs[1]
                entityAllNames=[" ".join(cname.split("_")) for cname in segs[2].split("||")]
                entityID2entityPreferredName[entityID]=entityPreferredName
                entityID2AllNames[entityID]=entityAllNames
    
    return entityID2entityPreferredName, entityID2AllNames


def name2id(file_path):
    """Returns the lookup between entityName and entityID.
    """
    
    entityID2entityPreferredName, entityID2AllNames=read_vocab(file_path)
    entityName2ID={}
    for key,value in entityID2AllNames.items():
        for t in value:
            t="_".join(t.split(" "))
            entityName2ID[t]=key
    
    return entityName2ID


def read_predict_pairs(vocab_path, predict_pairs_path):
    """Returns matching between predict pairs names and their corresponding ID.
    """
    entityName2ID=name2id(vocab_path)
    
    predict_pairs=[]
    with open(predict_pairs_path,"r") as f_in:
        for line in f_in:
            line=line.strip()
            if line:
                segs=line.split('\t')
                predict_pairs.append([entityName2ID[segs[0]], entityName2ID[segs[1]], int(segs[2])])
                
    return predict_pairs

def writer(filename, prediction):
    path = "../Results/" + filename + "_result.txt"
    with open(path, "w", encoding="utf-8") as f, open("./data/test_pairs.txt", "r", encoding="utf-8") as ref:
        f.write("hyper\thypo\tsim\n")
        for (word_line,pred) in zip(ref,prediction):
            w = word_line.split()
            f.write(w[0]+"\t")
            f.write(w[1]+"\t")
            f.write(str(pred))
            f.write("\n")
    f.close()
    ref.close()

def write_model(model, filename):
    path = "./data/"+filename+".pkl"
    file = open(path, "wb")
    str = pickle.dumps(model)
    file.write(str)
    file.close()