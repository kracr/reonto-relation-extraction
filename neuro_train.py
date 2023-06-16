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
import ast
from models.factory import get_model
from sklearn.model_selection import train_test_split
try:
    from functools import reduce
except:
    pass
import matplotlib.pyplot as plt
from semanticsimilarity import similarity
#from hops import getpath
from ontologypath import getpaths,path
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()



def train():
    device_id = 0
    # 
    ontologyname="OAE"
    database="ADE"  #Biorel or ADE
    model_name = "CNN"
    data_folder = "data/"
    save_folder = "data/models/"

    model_params = "model_params.json"
    word_embeddings = "glove.6B.50d.txt"
    if(database=="Biorel"):
     train_set = "Biorel/mytrain.json"
     val_set = "Biorel/mydev.json"
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
 
    sentence=[]
    entity=[]
    output=[]
    entity2id={}
    if(database=="Biorel"):
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
             sentence.append(lst[1])
             entity.append(lst[2])
             entity.append(lst[5])
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


    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))


    graphs_to_indices = sp_models.to_indices
    if(database=="Biorel"):
     if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_Biorel
     elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask_and_entity_pair_Biorel   
     elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_Biorel
     elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_entity_pair_Biorel
     elif model_name == "GPGNN_ONTOLOGY":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_entity_pair_Biorel
     elif model_name == "GPGNN_ONTOLOGY_BERT":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_entity_pair_Biorel
     _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)
    elif(database=="ADE"):
     if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_ADE
     elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask_and_entity_pair_ADE   
     elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_ADE
     elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_FORNEW
     elif model_name == "GPGNN_ONTOLOGY":
        graphs_to_indices=sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_FORNEW
     elif model_name == "GPGNN_ONTOLOGY_BERT":
        graphs_to_indices=sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_FORNEW_BERT
        
     _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)
    #train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))

    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    val_data = None


    print("Save property dictionary.")
    with open(data_folder + "models/" + model_name + ".property2idx", 'w') as outfile:
        outfile.write(str(property2idx))

    print("Training the model")

    print("Initialize the model")
   
    model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out).cuda()


    loss_func = nn.CrossEntropyLoss(ignore_index=0).cuda()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=model_params['weight_decay'])

    indices = np.arange(train_as_indices[0].shape[0])

    step = 0
    for train_epoch in range(model_params['nb_epoch']):
        if(shuffle_data):
            np.random.shuffle(indices)
        f1 = 0
        prec=0
        recal=0
        for i in tqdm(range(int(train_as_indices[0].shape[0] / model_params['batch_size']))):
            opt.zero_grad()

            sentence_input = train_as_indices[0][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            entity_markers = train_as_indices[1][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            labels = train_as_indices[2][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            
            if model_name is "GPGNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
            
            elif model_name == "GPGNN_ONTOLOGY":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])

            elif model_name == "GPGNN_ONTOLOGY_BERT":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]],
                                Variable(torch.as_tensor(np.array(train_as_indices[5]).astype(int))).cuda(),id2word)

            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda(), 
                                Variable(torch.from_numpy(np.array(train_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).cuda())
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).cuda(),
                                Variable(torch.from_numpy(entity_markers.astype(int))).cuda())

            loss = loss_func(output, Variable(torch.from_numpy(labels.astype(int))).view(-1).cuda())

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            opt.step()
            
            
            if(model_name != "LSTM" and model_name != "PCNN" and model_name != "CNN"):
              entity_pairs = train_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
              entity_pairs = reduce(lambda x,y :x+y , entity_pairs)
            else:
              entity_pairs = train_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]    
              print(entity_pairs)


            prop=[]
            for propkeys in property2idx.keys():
                prop.append(propkeys)

            for entity_pair in entity_pairs:
             hop_path=getpaths(entity_pair,ontologyname)
             if hop_path is not None:
              print(hop_path)
              biasedrelation,biasedscore=similarity(hop_path,prop)
              
              biasedrelationid=(property2idx[biasedrelation])
              output=torch.transpose(output,0,1)
              output[biasedrelationid]=output[biasedrelationid]+biasedscore
              output=torch.transpose(output,0,1)
                        
            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()
            

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(predicted, labels, empty_label=p0_index)
            f1 += add_f1
        print("Training f1: ", f1 /
                (train_as_indices[0].shape[0] / model_params['batch_size']))

        val_f1 = 0
        
        for i in tqdm(range(int(val_as_indices[0].shape[0] / model_params['batch_size']))):
            sentence_input = val_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            entity_markers = val_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            labels = val_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            if model_name == "GPGNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda(), 
                                val_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda(), 
                                Variable(torch.from_numpy(np.array(val_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), volatile=True).cuda())        
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda())
            #print("OUTPUTTTTT")
            #print(output)
            if(model_name != "LSTM" and model_name != "PCNN" and model_name != "CNN"):
              entity_pairs = val_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
              entity_pairs = reduce(lambda x,y :x+y , entity_pairs)
            else:
              entity_pairs = val_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]    
              print(entity_pairs)
              

            for entity_pair in entity_pairs:
             hop_path=getpaths(entity_pair,ontologyname)
             if hop_path is not None:
              print(hop_path)
              biasedrelation,biasedscore=similarity(hop_path,prop)
              
              biasedrelationid=(property2idx[biasedrelation])
              output=torch.transpose(output,0,1)
              output[biasedrelationid]=output[biasedrelationid]+biasedscore
              output=torch.transpose(output,0,1)
                        

            _, predicted = torch.max(output, dim=1)
            #print("max predicted")
            #print(_)
            #print("PREDICTEDDDDDDDDDD")
            #print(predicted)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _,_, add_f1 = evaluation_utils.evaluate_instance_based(
                predicted, labels, empty_label=p0_index)
            val_f1 += add_f1
            

        print("Validation f1: ", val_f1 /
                (val_as_indices[0].shape[0] / model_params['batch_size']))
        # save model
        if (train_epoch % 5 == 0 and save_model):
            torch.save(model.state_dict(), "{0}{1}-{2}.out".format(save_folder, model_name, str(train_epoch)))

        step = step + 1

train()
