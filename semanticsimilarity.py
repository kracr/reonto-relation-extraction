from sentence_transformers import SentenceTransformer, util
import numpy as np


model = SentenceTransformer('dmis-lab/biosyn-biobert-bc2gn')


def similarity(path, relations):
 if path is not None:
  #print("relations")
  # print(relations)
  rel_embedding = model.encode(relations, convert_to_tensor=True)
  path_embedding = model.encode(path, convert_to_tensor=True)
  # compute similarity scores of two embeddings
  # top_k results to return
  top_k=1
  # compute similarity scores of the sentence with the corpus
  cos_scores = util.pytorch_cos_sim(path_embedding, rel_embedding)[0]
  # Sort the results in decreasing order and get the first top_k
  top_results = np.argpartition(-cos_scores.cpu(), range(top_k))[0:top_k]
  #print("Sentence:", relations, "\n")
  #print("Top", top_k, "most similar sentences in corpus:")
  for idx in top_results[0:top_k]:
    yh=0
    #print(relations[idx], "(Score: %.4f)" % (cos_scores[idx]))      
 
  #return relations[idx],idx 
  return relations[idx],cos_scores[idx]/10
 else:
  return "",0
