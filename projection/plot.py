#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt

Epoch = []
PosLoss = []
NegLoss = []
for line in open('./modLog.txt',"r",encoding="utf-8"):
    if line.split()[0]=='Epoch':
        continue
    else:
        Epoch.append(line.split()[0])
        PosLoss.append(line.split()[2])
        NegLoss.append(line.split()[3])

epoch = [float(i) for i in Epoch]
posloss = [float(i) for i in PosLoss]
negloss = [float(i) for i in NegLoss]
plt.plot(epoch, posloss, color='blue', label='PosLoss')
plt.plot(epoch, negloss, color='green', label='NegLoss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Positive/ Negative Pair Loss')
plt.legend()
plt.grid(True)
plt.savefig('./loss.png')

