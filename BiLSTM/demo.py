#!/usr/bin/env python
# coding: utf-8

# In[10]:


from evalu import *

with open("./data/res.score.txt", "w", encoding="utf-8") as f:
    pred_dev_ins, golden_dev_ins = test()
    # f1 score
    test_f1 = evaluate(golden_dev_ins, pred_dev_ins)
    f.write(str(test_f1)+"\n")
f.close()

with open("./data/test.txt", "r", encoding="utf-8") as f_read, open("./data/BiLSTM.test.txt", "w", encoding="utf-8") as f:
    sens = []; t = []
    for line in f_read:
        if line!='\n':
            t.append(line)
        else:
            sens.append(t)
            t=[]
    
    for sen,pred in zip(sens,pred_dev_ins):
        if len(sen) == len(pred):
            _ = []
            for i in range(len(sen)):
                if pred[i] != 'O':
                    _.append(sen[i].split()[0]+" "+pred[i])
            f.write(str(_)[1:-1])
            f.write("\n")
f_read.close()
f.close()

