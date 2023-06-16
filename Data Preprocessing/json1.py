import json
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from time import time
import argparse
import os
import pandas as pd
import numpy as np
import networkx as nx
import nltk
import tensorflow as tf
from tensorflow import keras

f = open('./data/data/test.json')
data = json.load(f)
data=data
entity1={}
entity2={}
sentence={}
relation={}
lexical_features0={}
lexical_features1={}
lexical_features2={}
syntactic_feature={}
data_x=[]
data_y=[]
features={}
#tokens={}
pattern=re.compile('syntactic_feature')
mygraph={}
#mygraph = {'edgeSet':['left':{},'right':{},'relation':{}],'head_CUI':{}],'tail_CUI':{},'tokens':{}],"vertexSet":['head':{},'tail':{},'relation':{}]}
syntactic_features={}
count=0
filename='./data/data/testjs1.json'
#with open(filename, mode='w', encoding='utf-8') as f:
#    json.dump([], f)


for i in data:
  
  #mygraph['vertexSet']=
  #mygraph['vertexSet']['tail']={}
  #mygraph['vertexSet']['relation']={}
  #mygraph['edgeSet']={}
  #mygraph['edgeSet']['right']={}
  #mygraph['edgeSet']['relation']={}
  #mygraph['edgeSet']['head_CUI']={}
  #mygraph['edgeSet']['tail_CUI']={}
  #mygraph['tokens']={}
  #mygraph = {'edgeSet':['left':{},'right':{},'relation':{},'head_CUI':{},'tail_CUI':{}],'tokens':[],"vertexSet":['head':{},'tail':{},'relation':{}]}
   #i['head'].append('kbid',head)
   #i['tail'].append('kbid',tail)
   feed=[]
   head={}
   tail={}
   head=i['head']['CUI']
   tail=i['tail']['CUI']
   mygraph={'vertexSet':[],'edgeSet':[],'tokens':[]}
   tokens=nltk.word_tokenize(i["sentence"])
   #i["head"]["kbid"] = i.pop(['head']['CUI'])
   #i["tail"]["kbid"]=i.pop(['tail']['CUI'])
   #json_data = i.replace('key_old1','Key_new1').replace('key_old2','key_new2')
   #mygraph["vertexSet"].append({'kbid':i["head"]["CUI"]})
   mygraph["vertexSet"].append({'head':i['head'],'tail':i['tail'],'relation':i['relation']})
   #mygraph["vertexSet"].append({'kbid':i["tail"]["CUI"]})
   #mygraph["edgeSet"].append({'left':i['head']['split_start'],'right':i['tail']['split_start'],'relaion':i['relation'],'head_CUI':i['head']['CUI'],'tail_CUI':i['tail']['CUI']})
   mygraph['tokens'].append(tokens)
   #mygraph=dict(zip(mygraph['vertexSet'], mygraph['edgeSet'],mygraph['tokens']))
   feed.append(mygraph)
   
   
   with open(filename, 'a') as outfile:
    outfile.write(json.dumps(feed,indent=2))
    outfile.write(",")
    outfile.close()

   #with open(filename, mode='w', encoding='utf-8') as feedsjson:
   # feeds.append(mygraph)
   # json.dump(feeds, feedsjson)   
#with open(filename, 'a') as fp:
# json.dump(mygraph,fp)
   #fp.write('\n')
        #json.dump(mygraph,fp)
        #json.dump(mygraph['edgeSet'],fp)
        #json.dump(mygraph['tokens'],fp)
        #fp.write('\n')
   #json.dump(mygraph,fp)
   #fp.write('\n') 
   #json.dump(mygraph['edgeSet'],fp)
   #fp.write('\n') 
   #json.dump(mygraph['tokens'],fp)
   #fp.write('\n')
  #out_file = open('./data/trainj.json','a')
  #json.dump(mygraph,out_file) 
""" 
  entity1[count]=i['head']['word']
  entity2[count]=i['tail']['word']
  sentence[count]=i['sentence']
  relation[count]=i['relation']
  #print(relation[count])
  data_y.append(relation[count])
  lexical_features0[count]=i['lexical_feature0']
  lexical_features1[count]=i['lexical_feature1']
  lexical_features2[count]=i['lexical_feature2']
  
  class_values = sorted(relation[count].unique())
  class_idx = {name: id for id, name in enumerate(class_values)}
  features_idx = {name: idx for idx, name in enumerate(sorted(lexical_features0[count].unique()))}
  
  data_x.append(entity1[count]+", "+entity2[count])
  data_x["source"] = entity1[count].apply(lambda name: features_idx[name])
  data_x["target"] = entity2[count].apply(lambda name: features_idx[name])
  data_y["subject"] = relation[count].apply(lambda value: class_idx[value])
 
  #x=re.findall(r"([^.]*?syntactic_feature[^.]*\.)",str(i))
  #if(len(x)>0):
  # syntactic_feature[count]=x

  #print(x)
#relation id extraction
#datay=[]
#f1 = open(biorel+'relation2id.json')
#reldata = json.load(f1)
#for i in data_y:
    #print(i)
 #   datay.append(reldata[i])
 
from sklearn.model_selection import train_test_split     
from keras.preprocessing.text import Tokenizer  
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data_x)

X_train = tokenizer.texts_to_sequences(data_x)
#X_test = tokenizer.texts_to_sequences(sentences_test)
# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1                          

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

#X_test = pad_sequences(X_test, padding='post', maxlen=maxlen


import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50
embedding_matrix = create_embedding_matrix(biorel+'glove.6B.50d.txt/glove.6B.50d.txt' ,tokenizer.word_index, embedding_dim)
  
f.close()
"""
