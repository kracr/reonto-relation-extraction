import spacy 
import ast
import en_ner_bionlp13cg_md
nlp=spacy.load('en_ner_bionlp13cg_md')
for line in open("ADETEXT.txt").read().split('\n'):
    if(len(line))>0:
       edg={}
       edgeSet=[]
       edg["left"]=[]
       edg["right"]=[]
       stripline=(line).strip().split("|")
       print(stripline)
       print(len(stripline))
       if(len(line)>0):
         eid,sen,e1,len1,len2,e2,l1,l2,rel = (line).strip().split("|")


property2idx={"ADE":1,"NEGADE":0}
for line in open("ADETEXT.txt").read().split('\n'):
    if(len(line))>0:
       edg={}
       edgeSet=[]
       edg["left"]=[]
       edg["right"]=[]
       if(len(line)>0):
         eid,sen,e1,len1,len2,e2,l1,l2,rel = (line).strip().split("|")

         parsed_sentence = nlp(sen) 
         left=0
         right=0
         for token in parsed_sentence:
          if(e1==token.text):
            left=token.i
          elif(e2==token.text):
            right=token.i
         kbid=property2idx[rel]
         edg["kbID"]=kbid
         edg["left"].append(left)
         edg["right"].append(right)
          #edg["right"]=right
          #edg["left"]=left
         edgeSet.append(edg)         
         with open ("data/ADE-Corpus-V2/ADE19DEC.txt","a+") as f:
           final=line+"|"+str(edgeSet)
           f.write(final)
           f.write("\n")

