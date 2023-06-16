import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5
from simpletransformers.t5 import T5Model, T5Args
from utils import evaluation_utils, embedding_utils
from semanticgraph import io
from parsing import legacy_sp_models as sp_models
from models import baselines
import numpy as np
from sacred import Experiment
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import logging
import ast
from models.factory import get_model
from sklearn.model_selection import train_test_split
try:
    from functools import reduce
except:
    pass
import matplotlib.pyplot as plt
#from semanticsimilarity import similarity
#from hops import getpath
#from ontologypath import getpaths,path
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()
import pandas as pd

dfs = [] 


def train():
    device_id = 0
    # 
    ontologyname="NCI"
    database="Biorel"  #Biorel or ADE
    model_name = "CNN"
    data_folder = "data/"
    save_folder = "data/models/"
    lab=[]
    model_params = "model_params.json"
    word_embeddings = "glove.6B.50d.txt"
    if(database=="Biorel"):
     train_set = "mytrain.json"
     val_set = "mydev.json"
    elif(database=="ADE"):
     train_set = "ADE-Corpus-V2/ADE.txt"
     val_set = "ADE-Corpus-V2/ADE.txt"
    # a file to store property2idx
    # if is None use model_name.property2idx
    #property_index = "relationid.txt"
    property_index=None
    #entity2id="entity.txt"
    learning_rate = 1e-3
    shuffle_data = True
    save_model = True
    grad_clip = 0.25
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(data_folder + word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    if(database=="Biorel"):
     training_data, _ = io.load_relation_graphs_from_file(data_folder + train_set, load_vertices=True)
     val_data, _ = io.load_relation_graphs_from_file(data_folder + val_set, load_vertices=True)
    elif(database=="ADE"):
     training_data, _ = io.load_relation_graphs_from_file_ADE(data_folder + train_set, load_vertices=True)
     training_data, val_data = train_test_split(training_data, test_size=0.5, shuffle=True)
     val_data,test_data=train_test_split(val_data,test_size=0.5,shuffle=True)
     with open("ADEtestdata.txt","w") as testf:
      for line in test_data:
       testf.write(str(line))
       testf.write("\n")
    prefix=[] 
    sentence=[]
    entity=[]
    output=[]
    entity2id={}
    if(database=="Biorel"):
     for g in training_data:
        sen=g["sentence"]+g["head"]["word"]+g["tail"]["word"]
        sentence.append(sen)
        lab.append(g["relation"])
     if property_index:
        print("Reading the property index from parameter")
        with open(data_folder + "relationid.txt") as f:
            property2idx = ast.literal_eval(f.read())
     else:
        _, property2idx = embedding_utils.init_random({g["relation"] for g in training_data}, 1, add_all_zeroes=True, add_unknown=True)
        #_, property2idx = embedding_utils.init_random({g["relation"] for g in val_data}, 1, add_all_zeroes=True, add_unknown=True)
    elif(database=="ADE"):
     for line in open("./data/ADE-Corpus-V2/ADE.txt").read().split('\n'):
          if(len(line))>0:
             lst = line.split('|')
             #print(lst)
             sentence.append(str(lst[1]))
             entity.append(lst[2])
             entity.append(lst[5])
             prefix.append("relation extraction")
             lab.append(str(lst[8]))
             _, property2idx = embedding_utils.init_random({'ADE',"NEGADE"}, 1, add_all_zeroes=True, add_unknown=True)

     #print(property2idx)
     i=0
     for x in entity:
       if x not in output:
         output.append(x)
     for y in entity:
         entity2id[y]=i
         i=i+1
         #print(y)    
    
     with open ("entity2id.txt","w") as frr:
      frr.write(json.dumps(entity2id))
    if(database=="ADE"):
      max_sent_len = max(len(sen) for sen in sentence)
      print("Max sentence length:", max_sent_len) 

    elif(database=="Biorel"):  
      max_sent_len = max(len(g["sentence"]) for g in training_data)
      print("Max sentence length:", max_sent_len)   
 
    import pandas as pd
  
    # create an Empty DataFrame object
    df = pd.DataFrame()
    df['prefix']=(prefix)
    df['input_text']=(sentence)
    df['target_text']=(lab) 
    #print(df)
    #for g in training_data:
    #  df['source_text'] =pd.DataFrame(g["sentence"])
    #  df['target_text']=pd.DataFrame(g["relation"])

      #df = pd.DataFrame.from_dict(g, orient ='index')
      #df = pd.DataFrame(g["sentence"])
      #dfs.append(df)  # append each dataframe to the list
      #print(df)
      #df2=pd.DataFrame(g["relation"])
      #testdf.append(df2)
      #print(testdf)
    train_df, test_df = train_test_split(df, test_size=0.2)
    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))


    graphs_to_indices = sp_models.to_indices
    if(database=="Biorel"):
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_entity_pair_Biorel
        _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)
    elif(database=="ADE"):
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_FORNEW
        _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)
    #train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))

    #training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    val_data = None
    """
    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")
    model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, max_epochs=3, use_gpu=True)

    result = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score,f1=sklearn.metrics.f1_score)
    """
    model_args = T5Args()
    model_args.num_train_epochs = 200
    model_args.no_save = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.overwrite_output_dir = True

    model = T5Model("t5", "t5-base", args=model_args)

    model.train_model(train_df,eval_data=test_df,acc=sklearn.metrics.accuracy_score)

    # Evaluate the model
    #result = model.eval_model(test_df,acc=sklearn.metrics.accuracy_score,f1=sklearn.metrics.f1_score)
train()
