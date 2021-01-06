#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def reader(filepath):
    '''Mainly read in txt file
    '''
    with open(filepath,'r',encoding='utf-8') as f:
        data=f.read()
    return data

def writer(filepath,pair_list):
    '''Write hypernymy list into txt file
    '''
    file=open(filepath,'w+',encoding='utf-8')
    for item in pair_list:
        file.write(str(item))
        file.write('\n')
    file.close()


# def train_test_split(data,train_test_ratio=1):
#     index=np.random.permutation(range(len(self.data)))  # may reverse to a valid set
#     shuffle=[]
#     for i in index: shuffle.append(self.data[i])
#     train=shuffle[:int(np.round(len(self.data)*train_test_ratio))]
#     test=shuffle[int(np.round(len(self.data)*train_test_ratio)):]
#     return train,test

