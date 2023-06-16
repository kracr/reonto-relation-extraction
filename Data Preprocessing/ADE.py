from datasets import load_dataset
import numpy as np
import pandas as pd
import scispacy
import spacy
from spacy import displacy
import en_ner_bionlp13cg_md
import re

dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")
print(dataset)
df = pd.DataFrame(dataset['train'])
df.sample(5, random_state=124)

train, val, test = np.split(df.sample(frac=1), [int(0.7*len(df)), int(0.9*len(df))])
#1df.to_csv(r'pandas.txt', header=None, index=None, sep='|', mode='a')


nlp = spacy.load("en_ner_bionlp13cg_md")   #Define the pre-trained model.


for line in open("pandas.txt","r"):
   sen,label=line.split('|')
   #print(str(label))
   labelled=""
   if("1" in str(label)):
    labelled="ADE"
   elif("0" in str(label)):
    labelled="NEGADE"
   print(labelled)
   doc=nlp(sen)
   tokens = [token.text for token in doc] 
   #print(tokens)
   #for index,ele in enumerate(tokens):
   #  print(index)
   #print(doc.ents[0])
   #print(doc.text)
   #print(len(doc.ents))


   if(len(doc.ents)==2):
      #print(doc.text)
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[1])
      count=0
      index0=0
      index1=0
      ele0=""
      ele1=""
      strlabel=""
      for index,ele in enumerate(tokens):
            if str(ele)== str(doc.ents[0]):
               ele0=doc.ents[0]
               index0=index
            if(str(ele)==str(doc.ents[1])):
               ele1=doc.ents[1]
               index1=index
            #print(label)
            if(label=="0"):
              strlabel="AE"
            if(label)=="1":
              strlabel="negAE"
      final=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele1)+"|0|1|"+labelled
      #print(final)
      with open ("ADETEXT.txt","a+") as f:
           f.write(final)
           f.write("\n")
   if(len(doc.ents)==3):
      #print(doc.text)
      #print("Entity1:",doc.ents[0],"Entity2:",doc.ents[1])
      count=0
      index0=0
      index1=0
      index2=0
      ele0=""
      ele1=""
      ele2=""
      strlabel=""
      for index,ele in enumerate(tokens):
            if str(ele)== str(doc.ents[0]):
               ele0=doc.ents[0]
               index0=index
            if(str(ele)==str(doc.ents[1])):
               ele1=doc.ents[1]
               index1=index
            if(str(ele)==str(doc.ents[2])):
              ele2=doc.ents[2]
              index2=index
            if(label=="0"):
              strlabel="AE"
            if(label)=="1":
              strlabel="negAE"
      with open("ADETEXT.txt","a+") as f:
    
       final1=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele1)+"|0|1|"+labelled
       f.write("\n")
       final2=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele2)+"|0|1|"+labelled
       f.write("\n")
       final3=str(count)+"|"+str(doc.text)+"|"+str(ele1)+"|0|1|"+str(ele2)+"|0|1|"+labelled
       f.write("\n")
      #print(final)
      with open ("testADETEXT.txt","a+") as f:
           f.write(final1)
           f.write("\n")
           f.write(final2)
           f.write("\n")
           f.write(final3)
           f.write("\n")
   if(len(doc.ents)==4):
      #print(doc.text)
      #print("Entity1:",doc.ents[0],"Entity2:",doc.ents[1])
      count=0
      index0=0
      index1=0
      index2=0
      index3=0
      ele0=""
      ele1=""
      ele2=""
      ele3=""
      strlabel=""
      for index,ele in enumerate(tokens):
            if str(ele)== str(doc.ents[0]):
               ele0=doc.ents[0]
               index0=index
            if(str(ele)==str(doc.ents[1])):
               ele1=doc.ents[1]
               index1=index
            if(str(ele)==str(doc.ents[2])):
              ele2=doc.ents[2]
              index2=index
            if(str(ele)==str(doc.ents[3])):
               ele3=doc.ents[3]
               index3=index
            if(label=="0"):
              strlabel="AE"
            if(label)=="1":
              strlabel="negAE"
      final1=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele1)+"|0|1|"+labelled
      final2=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele2)+"|0|1|"+labelled
      final3=str(count)+"|"+str(doc.text)+"|"+str(ele0)+"|0|1|"+str(ele2)+"|0|3|"+labelled
      final4=str(count)+"|"+str(doc.text)+"|"+str(ele1)+"|0|1|"+str(ele2)+"|0|1|"+labelled
      final5=str(count)+"|"+str(doc.text)+"|"+str(ele1)+"|0|1|"+str(ele3)+"|0|1|"+labelled
      final6=str(count)+"|"+str(doc.text)+"|"+str(ele2)+"|0|1|"+str(ele3)+"|0|1|"+labelled
      #print(final)
      with open ("ADETEXT.txt","a+") as f:
           f.write(final1)
           f.write("\n")
           f.write(final2)
           f.write("\n")
           f.write(final3)
           f.write("\n")
           f.write(final4)
           f.write("\n")
           f.write(final5)
           f.write("\n")
           f.write(final6)
           f.write("\n")
                 
   """ 
   if(len(doc.ents)==3):
      "ENTITY#######################333333333333333333333333"
      print(doc.text)
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[1])
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[2])
      print("Entity1:",doc.ents[1],"Entity2:",doc.ents[2])
   if(len(doc.ents)==4):
      print(doc.text)
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[1])
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[2])
      print("Entity1:",doc.ents[0],"Entity2:",doc.ents[3])
      print("Entity1:",doc.ents[1],"Entity2:",doc.ents[2])
      print("Entity1:",doc.ents[1],"Entity2:",doc.ents[3])
      print("Entity1:",doc.ents[2],"Entity2:",doc.ents[3])
   """
      
"""
   if(len(line))>0:
             lst = line.split('|')
             #print(lst[0])
             doc=nlp(lst[0])
             #print(doc.ents)            
             #print("len",len(doc.ents))
             if(len(doc.ents))==1:
                for token in doc:
                  #print(token.text)
                  entity1=token.text[0]  
                  left=0
                for index,senentity in enumerate(lst[0].split(" ")):
                 for ent in doc.ents:
                  if str(ent)==str(senentity):
                     right=index
                     entity2=senentity
                print(left,right,entity1,entity2)  
             #for ent in doc.ents:
             #   for index,senentity in enumerate(lst[1].split(" ")):
             #        if str(ent)==str(senentity): 
             #            print(index,senentity)
               

            #  print(len(ent))
            #sentence.append(lst[1])
            #entity.append(lst[2])
            #entity.append(lst[5])
            #edg["kbID"]=kbid
            #edg["left"].append(left)
            #edg["right"].append(right)
            #edg["right"]=right
            #edg["left"]=left
            #edgeSet.append(edg)         
            #with open ("newfileADE.txt","a+") as f:
            # final=line+"|"+str(edgeSet)
            # f.write(final)
            # f.write("\n")



"""
