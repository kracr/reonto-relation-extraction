import rdflib
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace, RDF
import re
import timeit
import time


def NCIone(c1,c2,rel):
  
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1=" SELECT  ?pred1  WHERE  {<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c1+">    ?pred1  "
  re2=" <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c2+">}  "
  path=""  
  res_new=re1+re2
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      print(prop1,row["obj"])
      #prop2=row["pred2"].rsplit('/', 1)[-1]
      #prop3=row["pred3"].rsplit('/', 1)[-1]
      #final="Entity:"+c1+c2+"s:"+row["pred1"]+"pred:"+row["pred2"]
     path=str(prop1)
  return path

def NCItwo(c1,c2,rel):
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1=" SELECT  ?pred1 ?b ?pred2 WHERE  {<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c1+">    ?pred1   ?b. ?b  ?pred2 <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c2+">}  "
  path=""  
  res_new=re1
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["b"].rsplit('/', 1)[-1]
      prop3=row["pred2"].rsplit('/', 1)[-1]
      #final="Entity:"+c1+c2+"s:"+row["pred1"]+"pred:"+row["pred2"]
     path=str(prop1)+str(prop2)+str(prop3)
  return path



def NCIthree(c1,c2,rel):
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1=" SELECT  ?pred1 ?b ?pred2  ?c ?pred3 WHERE  {<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c1+">    ?pred1   ?b.  ?b   ?pred2  ?c. ?c  ?pred3 <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c2+">}  "
  path=""  
  res_new=re1
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):


      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["b"].rsplit('/', 1)[-1]
      prop3=row["pred2"].rsplit('/', 1)[-1]
      prop4=row["c"].rsplit('/', 1)[-1]
      prop5=row["pred3"].rsplit('/', 1)[-1]
      #final="Entity:"+c1+c2+"s:"+row["pred1"]+"pred:"+row["pred2"]
     path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)

  return path



def NCIaxiom1(c1,c2,g1):
  
     query="Select ?pred1 ?b  ?d ?e ?f  where{{?a  ?pred1  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c1+">} {?a ?b  _:c} { _:c  ?d ?e} {?e  ?f <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c2+"> } }"
     print(query)
     result=g1.query(query)
     path=""
     for row in result:
        if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["b"].rsplit('/', 1)[-1]
          prop3=row["d"].rsplit('/', 1)[-1]
          prop4=row["e"].rsplit('/', 1)[-1]
          prop5=row["f"].rsplit('/', 1)[-1]
          
        path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)
      
        return path



def NCIaxiom2(c1,c2,g1):
  
     query="Select ?pred1 ?a ?b  ?d ?e ?f ?g  where{ {?a  ?pred1  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c1+">} {?a ?b  _:c} { _:c  ?d ?e} {?e   ?f   ?g} {?g  ?h <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+c2+"> } }"
     print(query)
     result=g1.query(query)
     path=""
     for row in result:
        if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["a"].rsplit('/', 1)[-1]
          prop3=row["b"].rsplit('/', 1)[-1]
          prop4=row["d"].rsplit('/', 1)[-1]
          prop5=row["e"].rsplit('/', 1)[-1]
          prop6=row["f"].rsplit('/', 1)[-1]
          prop7=row["g"].rsplit('/', 1)[-1]

        path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)+str(prop6)+str(prop7)
      
        return path

def convert(prop):
  #print(prop)
   prop=prop.replace("AQ","actual outcome")
   prop=prop.replace("QB","has actual outcome")
   prop=prop.replace("RQ","classifies")
   prop=prop.replace("RO","measures")
   prop=prop.replace("SY","same as")
   prop=prop.replace("CHD","is a ,part of")
   prop=prop.replace("PAR","inverse is a ,has part")
   prop=prop.replace("SIB", "sibling is in a")
   prop=prop.replace("RN","tradename of")
   prop=prop.replace("RB","has tradesname")

   return prop




def medline_direct(c1,c2,g1):
   start_time = timeit.default_timer()
   res_direct= """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1 
     WHERE { MEDLINE:"""+c1+"""    ?pred1  MEDLINE:"""+c2+""" .
     }"""
   prop1=""
   direct_res=g1.query(res_direct)
   for row in direct_res:
    prop1=row["pred1"].rsplit('/', 1)[-1]
    print("direct")
    print(timeit.default_timer() - start_time)
   return convert(prop1)

def medline_hop1(c1,c2,g1):
   start_time = timeit.default_timer()
   res_medline = """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1  ?c   ?pred2
     WHERE { MEDLINE:"""+c1+"""    ?pred1      ?c.  ?c   ?pred2  MEDLINE:"""+c2+""" .
     }"""
   prop1=""
   res_med = g1.query(res_medline)
   for row in res_med:
             #print(f"{row.pred1} {row.c} {row.pred2}")
             pred1=row["pred1"].rsplit('/', 1)[-1]
             c=row["c"].rsplit('/', 1)[-1]
             pred2=row["pred2"].rsplit('/', 1)[-1]
             #final="Entity1: "+head+"Entity2: "+tail+"Actual relation: "+rel+"predicate1: "+row["pred1"]+"predicate2: "+row["pred2"]
             prop1=str(convert(pred1))+" "+str(medline_label(c))+" "+str(convert(pred2))
   print(timeit.default_timer() - start_time, "hop1 seconds")
   return prop1

def medline_label(c1):
   query="SELECT ?o where{ <http://purl.bioontology.org/ontology/MEDLINEPLUS/"+c1+">  <http://www.w3.org/2004/02/skos/core#prefLabel> ?o}"  
   queryres=g1.query(query)
   for row in queryres:
     return row["o"]
    

def medline_hop2(c1,c2,g1):
    start_time = timeit.default_timer()
    res_medline = """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1  ?c   ?pred2 ?d ?pred3
     WHERE { MEDLINE:"""+c1+"""    ?pred1      ?c.  ?c  ?pred2 ?d.  ?d   ?pred3  MEDLINE:"""+c2+""" .
     }"""
     
    
    res_med = g1.query(res_medline)
    path=""
    for row in res_med:
             path=""
             #print(f"{row.pred1} {row.c} {row.pred2}")
             #query="SELECT ?o where{ <http://purl.bioontology.org/ontology/MEDLINEPLUS/"+c1+">  <http://www.w3.org/2004/02/skos/core#prefLabel> ?o}"  
             #queryres=g1.query(query)
             #for row1 in queryres:
             #  print(row1["o"])
             prop1=row["pred1"].rsplit('/', 1)[-1]
             prop2=row["c"].rsplit('/', 1)[-1]
             prop3=row["pred2"].rsplit('/', 1)[-1]
             prop4=row["d"].rsplit('/', 1)[-1]
             prop5=row["pred3"].rsplit('/', 1)[-1]
             path=(str(convert(prop1))+" "+str(medline_label(prop2))+" "+str(convert(prop3))+" "+str(medline_label(prop4))+" "+str(convert(prop5)))
    print(timeit.default_timer() - start_time, "hop2 seconds")
    return path



def medline_hop3(c1,c2,g1):
    start_time = timeit.default_timer()
    res_medline = """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1  ?c   ?pred2 ?d ?pred3 ?e ?pred4
     WHERE { MEDLINE:"""+c1+"""    ?pred1      ?c.  ?c  ?pred2 ?d. ?d  ?pred3  ?e. ?e   ?pred4  MEDLINE:"""+c2+""" .
     }"""
     
    
    res_med = g1.query(res_medline)
    path=""
    for row in res_med:
             path=""
             #print(f"{row.pred1} {row.c} {row.pred2}")
             #query="SELECT ?o where{ <http://purl.bioontology.org/ontology/MEDLINEPLUS/"+c1+">  <http://www.w3.org/2004/02/skos/core#prefLabel> ?o}"  
             #queryres=g1.query(query)
             #for row1 in queryres:
             #  print(row1["o"])
             prop1=row["pred1"].rsplit('/', 1)[-1]
             prop2=row["c"].rsplit('/', 1)[-1]
             prop3=row["pred2"].rsplit('/', 1)[-1]
             prop4=row["d"].rsplit('/', 1)[-1]
             prop5=row["pred3"].rsplit('/', 1)[-1]
             prop6=row["e"].rsplit('/', 1)[-1]
             prop7=row["pred4"].rsplit('/', 1)[-1]
             path=(str(convert(prop1))+" "+str(medline_label(prop2))+" "+str(convert(prop3))+" "+str(medline_label(prop4))+" "+str(convert(prop5))+" "+str(medline_label(prop6))+" "+str(convert(prop7)))
    print("hop3 :",timeit.default_timer() - start_time, "hop3 seconds")
    return path



def medline_axiom1(c1,c2,g1):
    start_time = timeit.default_timer()
    res_medline = """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1  ?a  ?b   ?d  ?e  ?pred4
     WHERE {{ MEDLINE:"""+c1+"""    ?pred1      ?a} {?a ?b  _:c} { _:c  ?d ?e} {?e   ?pred4  MEDLINE:"""+c2+ "}}"
     
    
    res_med = g1.query(res_medline)
    path=""
    for row in res_med:
             path=""
             #print(f"{row.pred1} {row.c} {row.pred2}")
             #query="SELECT ?o where{ <http://purl.bioontology.org/ontology/MEDLINEPLUS/"+c1+">  <http://www.w3.org/2004/02/skos/core#prefLabel> ?o}"  
             #queryres=g1.query(query)
             #for row1 in queryres:
             #  print(row1["o"])
             prop1=row["pred1"].rsplit('/', 1)[-1]
             prop2=row["a"].rsplit('/', 1)[-1]
             prop3=row["b"].rsplit('/', 1)[-1]
             prop4=row["d"].rsplit('/', 1)[-1]
             prop5=row["e"].rsplit('/', 1)[-1]
             prop6=row["pred4"].rsplit('/', 1)[-1]
             
             path=(str(convert(prop1))+" "+str(medline_label(prop2))+" "+str(convert(prop3))+" "+str(convert(prop4))+" "+str(medline_label(prop5))+" "+str(convert(prop6)))
    print(timeit.default_timer() - start_time, "axiom1 seconds")
    return path


def medline_axiom2(c1,c2,g1):
    start_time = timeit.default_timer()
    res_medline = """
     PREFIX MEDLINE:  <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
     SELECT DISTINCT ?pred1  ?a  ?b   ?d  ?g  ?pred4
     WHERE {{ MEDLINE:"""+c1+"""    ?pred1      ?a} {?a ?b  _:c} { _:c  ?d ?e} {?e ?f ?g }{?g   ?pred4  MEDLINE:"""+c2+ "}}"
     
    
    res_med = g1.query(res_medline)
    path=""
    for row in res_med:
             path=""
             #print(f"{row.pred1} {row.c} {row.pred2}")
             #query="SELECT ?o where{ <http://purl.bioontology.org/ontology/MEDLINEPLUS/"+c1+">  <http://www.w3.org/2004/02/skos/core#prefLabel> ?o}"  
             #queryres=g1.query(query)
             #for row1 in queryres:
             #  print(row1["o"])
             prop1=row["pred1"].rsplit('/', 1)[-1]
             prop2=row["a"].rsplit('/', 1)[-1]
             prop3=row["b"].rsplit('/', 1)[-1]
             prop4=row["d"].rsplit('/', 1)[-1]
             prop5=row["e"].rsplit('/', 1)[-1]
             prop6=row["f"].rsplit('/', 1)[-1]
             prop7=row["g"].rsplit('/', 1)[-1]
             prop8=row["pred4"].rsplit('/', 1)[-1]
             
             path=(str(convert(prop1))+" "+str(medline_label(prop2))+" "+str(convert(prop3))+" "+str(convert(prop4))+" "+str(medline_label(prop5))+" "+str(convert(prop6))+str(medline_label(prop7))+str(convert(prop8)))
    print(timeit.default_timer() - start_time, "axiom2 seconds")
    return path

def NDFRTone(c1,c2,g1):
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1= "SELECT  ?pred  WHERE  {?s    <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>    "
  re2="'"
  re3= c1
  re4="'"
  re5= "^^xsd:string."
  re6= "?s  ?pred  ?b."
  re7="?b  <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>     "
  re8="'"
  re9=c2
  re10="'"
  re11="^^xsd:string."
  re12="}"

  res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12
  path=""  
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):
      prop1=row["pred"].rsplit('/', 1)[-1]
      print(prop1)
      
     path=str(prop1)
     return path


def NDFRTtwo(c1,c2,g1):
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1= "SELECT  ?pred1  ?b  ?pred2 ?c  WHERE  {?s    <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>    "
  re2="'"
  re3= c1
  re4="'"
  re5= "^^xsd:string."
  re6= "?s  ?pred1  ?b. ?b  ?pred2  ?c. "
  re7="?c  <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>     "
  re8="'"
  re9=c2
  re10="'"
  re11="^^xsd:string."
  re12="}"

  res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12
  path=""  
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["b"].rsplit('/', 1)[-1]
      prop3=row["pred2"].rsplit('/', 1)[-1]
      prop4=row["c"].rsplit('/', 1)[-1]
      path=str(prop1)+str(prop2)+str(prop3)+str(prop4)

      
     #path=str(prop1)
     return path
    

def NDFRTthree(c1,c2,g1):
  c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  final={}
  re1= "SELECT  ?pred1  ?b  ?pred2 ?c ?pred3  ?d  WHERE  {?s    <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>    "
  re2="'"
  re3= c1
  re4="'"
  re5= "^^xsd:string."
  re6= "?s  ?pred1  ?b. ?b  ?pred2  ?c.  ?c  ?pred3  ?d."
  re7="?d  <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>     "
  re8="'"
  re9=c2
  re10="'"
  re11="^^xsd:string."
  re12="}"

  res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12
  path=""  
  print(res_new)
  query_mesh=str(res_new)
  #print(reason)  
  res_n = g1.query(query_mesh)
  for row in res_n:
     if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["b"].rsplit('/', 1)[-1]
      prop3=row["pred2"].rsplit('/', 1)[-1]
      prop4=row["c"].rsplit('/', 1)[-1]
      prop5=row["pred3"].rsplit('/', 1)[-1]
      prop4=row["d"].rsplit('/', 1)[-1]
      

      path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)+str(prop6)

      
     #path=str(prop1)
     return path


def NDFRTaxiom1(c1,c2,g1):
     re1= "SELECT  ?pred1  ?b  ?pred2 ?c ?pred3  ?d  WHERE  {?a    <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>    "
     re2="'"
     re3= c1
     re4="'"
     re5= "^^xsd:string."
     re6= "{?a ?b  _:c} { _:c  ?d ?e}"
     re7="?e  <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>     "
     re8="'"
     re9=c2
     re10="'"
     re11="^^xsd:string."
     re12="}"
  
     res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12
    
     result=g1.query(res_new)
     path=""
     for row in result:
        if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["b"].rsplit('/', 1)[-1]
          prop3=row["pred2"].rsplit('/', 1)[-1]
          prop4=row["c"].rsplit('/', 1)[-1]
          prop5=row["pred3"].rsplit('/', 1)[-1]
          prop6=row["d"].rsplit('/', 1)[-1]
        
          
        path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)+str(prop6)
      
        return path


def NDFRTaxiom2(c1,c2,g1):
     re1= "SELECT  ?b  ?d ?e ?f  ?g  WHERE  {?a    <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>    "
     re2="'"
     re3= c1
     re4="'"
     re5= "^^xsd:string."
     re6= "{?a ?b  _:c} { _:c  ?d ?e} {?e ?f ?g}"
     re7="?g  <http://evs.nci.nih.gov/ftp1/NDF-RT/NDF-RT.owl#UMLS_CUI>     "
     re8="'"
     re9=c2
     re10="'"
     re11="^^xsd:string."
     re12="}"
  
     res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10+re11+re12
    
     result=g1.query(res_new)
     path=""
     for row in result:
        if(len(row)>0):
          prop1=row["b"].rsplit('/', 1)[-1]
          prop2=row["d"].rsplit('/', 1)[-1]
          prop3=row["e"].rsplit('/', 1)[-1]
          prop4=row["f"].rsplit('/', 1)[-1]
          prop5=row["g"].rsplit('/', 1)[-1]
        
          
        path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)
      
        return path




def dinto_one(c1,c2,g2):
  #c1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  #c2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
  startTime = time.time()   
  path=""
  sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  sub2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
   
  re5="SELECT ?x1 ?pred1 ?x2 WHERE{{?s1   <http://www.w3.org/2000/01/rdf-schema#label>  ?x1 }"   
  re6= "{?s1  ?pred1  ?b}."
  re7="{?b  <http://www.w3.org/2000/01/rdf-schema#label>  ?x2}"
  re8=" FILTER(regex(str(?x1),"
  re9='"'
  re10= str(sub1)+'"'
  re11=") && regex(str(?x2),"
  re12='"'
  re13=str(sub2)+'"'
  re14="))}"

  query_mesh=re5+re6+re7+re8+re9+re10+re11+re12+re13+re14
  print(query_mesh)  

  #print(reason)  
  res_n = g2.query(query_mesh)
  executionTime = (time.time() - startTime)
  print('Execution time in dinto one seconds: ' + str(executionTime))
  for row in res_n:
    if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      path=str(prop1)
    return path
   

def dinto_two(c1,c2,g2):
  startTime = time.time()
  path=""
  sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  sub2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
  re5="SELECT ?x1 ?pred1 ?pred2 ?c WHERE{{?s1   <http://www.w3.org/2000/01/rdf-schema#label>  ?x1 }"   
  re6= "{?s1  ?pred1  ?b}. {?b  ?pred2  ?c}"
  re7="{?c  <http://www.w3.org/2000/01/rdf-schema#label>  ?x2}"
  re8=" FILTER(regex(str(?x1),"
  re9='"'
  re10= str(sub1)+'"'
  re11=") && regex(str(?x2),"
  re12='"'
  re13=str(sub2)+'"'
  re14="))}"

  query_mesh=re5+re6+re7+re8+re9+re10+re11+re12+re13+re14
  print(query_mesh)  

  #print(reason)  
  res_n = g2.query(query_mesh)
  executionTime = (time.time() - startTime)
  print('Execution time in DINTO two seconds: ' + str(executionTime))
  for row in res_n:
    if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["pred2"].rsplit('/',1)[-1]
      path=str(prop1)+str(row["c"])+str(prop2)
    return path

def dinto_three(c1,c2,g2):
  startTime = time.time()
  path=""
  sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
  sub2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
  re5="SELECT  ?pred1 ?pred2  ?pred3 ?c  ?d ?x2 WHERE{{?s1   <http://www.w3.org/2000/01/rdf-schema#label>  ?x1 }"   
  re6= "{?s1  ?pred1  ?b}. {?b  ?pred2  ?c} {?c  ?pred3  ?d}"
  re7="{?d  <http://www.w3.org/2000/01/rdf-schema#label>  ?x2}"
  re8=" FILTER(regex(str(?x1),"
  re9='"'
  re10= str(sub1)+'"'
  re11=") && regex(str(?x2),"
  re12='"'
  re13=str(sub2)+'"'
  re14="))}"

  query_mesh=re5+re6+re7+re8+re9+re10+re11+re12+re13+re14
  #print(query_mesh)  

  res_n = g2.query(query_mesh)
  executionTime = (time.time() - startTime)
  print('Execution time in dinto three seconds: ' + str(executionTime))
  prop1=""
  prop2=""
  for row in res_n:
    if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["pred2"].rsplit('/', 1)[-1]
      prop3=row["pred3"].rsplit('/', 1)[-1]
    
      path=str(prop1)+str(row["b"])+str(prop2)+str(row["c"])+str(prop3)+str(row["d"])
    
    return path

def dinto_axiom1(c1,c2,g2):
     startTime = time.time()
     path=""
     sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
     sub2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
     q1="Select ?pred1 ?b  ?d  ?f  where{{?a  ?pred1  ?all1}{?a ?b  _:c} { _:c  ?d ?e} {?e  ?f  ?g }"
     q2="FILTER regex(?all1,'"
     q3=sub1+"') FILTER regex(?g,'"+sub2+"')}"
     query=q1+q2+q3
     #print(query)
     result=g1.query(query)
     executionTime = (time.time() - startTime)
     print('Execution time in dinto axiom1seconds: ' + str(executionTime))
     path=""
     for row in result:
         if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["b"].rsplit('/', 1)[-1]
          prop3=row["d"].rsplit('/', 1)[-1]
          prop4=row["f"].rsplit('/', 1)[-1]

         path=str(prop1)+str(prop2)+str(prop3)+str(prop4)
      
     return path


def dinto_axiom2(c1,c2,g2):
     startTime = time.time()
     path=""
     sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', c1)
     sub2 = re.sub(r'[^a-zA-Z0-9\s]', '', c2)
     q1="Select ?pred1 ?pred2 ?b ?d ?f   where{{?a  ?pred1  ?all0}{?a  ?pred2  ?all1}{?all1 ?b  _:c} { _:c  ?d ?e} {?e  ?f  ?g }"
     q2="FILTER regex(?all1,'"
     q3=sub1+"') FILTER regex(?g,'"+sub2+"')}"
     query=q1+q2+q3
     #print(query)
     result=g1.query(query)
     executionTime = (time.time() - startTime)
     print('Execution time in dinto axiom2 seconds: ' + str(executionTime))
     path=""
     for row in result:
         if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["pred2"].rsplit('/', 1)[-1]
          prop3=row["b"].rsplit('/', 1)[-1]
          prop4=row["d"].rsplit('/', 1)[-1]
          prop5=row["f"].rsplit('/', 1)[-1]

         path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)
      
     return path
def oae_path1(c1,c2,g1):
 path=""  
 q31="SELECT ?a ?b ?c ?d WHERE {{?a   ?p  "
 q32='"'
 q33=str(ent1)
 q34='"'
 q35="} {?a ?b  ?c}{?c  ?d  "
 q36='"'
 q37=str(ent2)
 q38='"}}'
 res_new=q31+q32+q33+q34+q35+q36+q37+q38
 query_mesh=str(res_new)
 res_n = g1.query(query_mesh)
 for row in res_n:
    if(len(row)>0):
      prop2=row["b"].rsplit('/', 1)[-1]
      prop3=row["d"].rsplit('/', 1)[-1]

      path=str(row["a"])+str(prop2)+str(row["c"])+str(prop3)
 return path

def oae_path2(c1,c2,g1):
 path="" 
 q31="SELECT ?a ?b ?c ?d ?e  ?f WHERE {{?a   ?p  "
 q32='"'
 q33=str(ent1)
 q34='"'
 q35="} {?a ?b  ?c} {?c  ?d  ?e}{?e  ?f  "
 q36='"'
 q37=str(ent2)
 q38='"}}'
 res_new=q31+q32+q33+q34+q35+q36+q37+q38
 query_mesh=str(res_new)
 res_n = g1.query(query_mesh)
 for row in res_n:
    if(len(row)>0):
      prop1=row["b"].rsplit('/', 1)[-1]
      prop2=row["d"].rsplit('/', 1)[-1]

      path=str(row["a"])+str(prop1)+str(row["c"])+str(prop2)+str(row["e"])
 return path


def oae_path3(c1,c2,g1):
 path=""
 q31="SELECT  ?b ?c ?d ?e ?f  ?g  ?h WHERE {{?a   ?p  "
 q32='"'
 q33=str(ent1)
 q34='"'
 q35="} {?a ?b  ?c} {?c  ?d  ?e} {?e ?f  ?g}{?g  ?h  "
 q36='"'
 q37=str(ent2)
 q38='"}}'
 res_new=q31+q32+q33+q34+q35+q36+q37+q38
 query_mesh=str(res_new)
 res_n = g1.query(query_mesh)
 for row in res_n:
    if(len(row)>0):
      prop1=row["b"].rsplit('/', 1)[-1]
      prop2=row["d"].rsplit('/', 1)[-1]
      prop3=row["f"].rsplit('/', 1)[-1]

      path=str(row["a"])+str(prop1)+str(row["c"])+str(prop2)+str(row["e"])+str(prop3)
 return path



def oae_axiom1(c1,c2,g1):
 path=""
 q31="SELECT ?b ?d ?e ?f WHERE {{?a   ?p  "
 q32='"'
 q33=str(ent1)
 q34='"'
 q35="} {?a ?b  _:c} { _:c  ?d ?e} {?e  ?f  "
 q36='"'
 q37=str(ent2)
 q38='"}}'
 res_new=q31+q32+q33+q34+q35+q36+q37+q38
 
 query_mesh=str(res_new)
 res_n = g1.query(query_mesh)
 for row in res_n:
    if(len(row)>0):
      prop1=row["b"].rsplit('/', 1)[-1]
      prop2=row["d"].rsplit('/', 1)[-1]
      prop3=row["f"].rsplit('/', 1)[-1]

      path=str(prop1)+str(prop2)+str(row["e"])+str(prop3)
 return path



def oae_axiom2(c1,c2,g1):
 path=""
 q31="SELECT ?b ?d ?e ?f ?g ?h WHERE {{?a   ?p  "
 q32='"'
 q33=str(ent1)
 q34='"'
 q35="} {?a ?b  _:c} { _:c  ?d ?e} {?e  ?f  ?g}{?g  ?h  "
 q36='"'
 q37=str(ent2)
 q38='"}}'
 res_new=q31+q32+q33+q34+q35+q36+q37+q38
 
 query_mesh=str(res_new)
 res_n = g1.query(query_mesh)
 for row in res_n:
    if(len(row)>0):
      prop1=row["b"].rsplit('/', 1)[-1]
      prop2=row["d"].rsplit('/', 1)[-1]
      prop3=row["f"].rsplit('/', 1)[-1]

      path=str(prop1)+str(prop2)+str(row["e"])+str(prop3)+str(row["g"])
      
 return path



def path(ontologyname,c1,c2,g1):
    path1=""
    path2=""
    path3=""
    pathaxiom1=""
    pathaxiom2=""
    # For Biorel dataset chose from MEDLINE, NCI, NDFRT
    # For ADE dataset chose from DINTO, OAE and DRON ontology
    if(ontologyname=="MEDLINE"):
       path1=medline_direct(c1,c2,g1)
       path2=medline_hop1(c1,c2,g1)
       path3=medline_hop2(c1,c2,g1)
       pathaxiom1=medline_axiom1(c1,c2,g1)
       pathaxiom2=medline_axiom2(c1,c2,g1)

    elif(ontologyname=="NCI"):

      path1=NCIone(c1,c2,g1)
      path2=NCItwo(c1,c2,g1)
      path3=NCIthree(c1,c2,g1)
      pathaxiom1=NCIaxiom1(c1,c2,g1)
      pathaxiom2=NCIaxiom2(c1,c2,g1)
      
    elif(ontologyname=="NDFRT"):
      
      path1=NDFRTone(c1,r2,g1)
      path2=NDFRTtwo(c1,c2,g1)
      path3=NDFRTthree(c1,c2,g1)
      pathaxiom1=NDFRTaxiom1(c1,c2,g1)
      pathaxiom2=NDFRTaxiom2(c1,c2,g1)
      

    elif(ontologyname=="DINTO"):
      

      path1=dinto_one(c1,c2,g1)
      path2=dinto_two(c1,c2,g1)
      path3=dinto_three(c1,c2,g1)
      pathaxiom1=dinto_axiom1(c1,c2,g1)
      pathaxiom2=dinto_axiom2(c1,c2,g1)

    elif(ontologyname=="OAE"):
      
      path1=oae_path1(c1,c2,g1)
      path2=oae_path2(c1,c2,g1)
      path3=oae_path3(c1,c2,g1)
      pathaxiom1=oae_axiom1(c1,c2,g1)
      pathaxiom2=oae_axiom2(c1,c2,g1)
    else:
      print("please enter correct ontology")
      
   
    return path1,path2,path3,pathaxiom1,pathaxiom2


def triplets(c1,c2,g1):
 query="SELECT ?s ?p ?o WHERE {?s ?p ?o}"
 result=g1.query(query)
 for res in result:
  print(res["s"],res["p"],res["o"])

ontologyname="DINTO"

if(ontologyname=="MEDLINE"):
  g1 = rdflib.Graph()
  g1.parse('./ontology/MEDLINEPLUS.ttl', format='ttl')

elif(ontologyname=="NCI"):
  g1 = rdflib.Graph()
  g1.parse("./ontology/NCI.owl")

elif(ontologyname=="NDFRT"):

  g1=rdflib.Graph()
  g1.parse("./ontology/NDF-RT.owl") 
elif(ontologyname=="OAE"):

  g1 = rdflib.Graph()
  g1 = g1.parse('./ontology/oae_merged.owl')

elif(ontologyname=="DINTO"):
   g1=rdflib.Graph()
   g1=g1.parse('./ontology/owlapi.xrdf',format="xml")

"""
with open("entpair.txt") as f:
 file =f.readlines()

 for line in file:
  fullline=line.split(",")
  for line in fullline:
   lst = line.strip(" ").split("|")
   if(len(lst))>0:
    number1 = re.sub(r'[^a-zA-Z0-9\s]', '', lst[0])
    number2 = re.sub(r'[^a-zA-Z0-9\s]', '', lst[1])
    print(number1,number2)
    NCIone(number1,number2,"")

#path1,path2,path3,pathaxiom1,pathaxiom2=path(ontologyname,"C0023819","C1333460",g1)
#final=str(path1)+"|"+str(path2)+"|"+str(path3)+"|"+str(pathaxiom1)+"|"+str(pathaxiom2)
#print("path1:",path1,"path2:",path2,"path3:",path3,"pathaxiom1",pathaxiom1,"pathaxiom2",pathaxiom2)


for line in open("./data/ADE-Corpus-V2/ADE.txt").read().split('\n'):
          if(len(line))>0:
             lst = line.split('|')
             #entity.append(lst[2])
             #entity.append(lst[5])
             path1=""
             path2=""
             path3=""
             pathaxiom1=""
             pathaxiom2=""
             path1,path2,path3,pathaxiom1,pathaxiom2=path(ontologyname,lst[2],lst[5],g1)
             final=str(path1)+"|"+str(path2)+"|"+str(path3)+"|"+str(pathaxiom1)+"|"+str(pathaxiom2)
             print("path1:",path1,"path2:",path2,"path3:",path3,"pathaxiom1",pathaxiom1,"pathaxiom2",pathaxiom2)
             
 
   


"""
def two(c1,c2,rel):
  re1= " SELECT  ?pred1 ?pred2 ?pred3  ?pred4 WHERE  {?a    ?pred1    "
  re2="'"
  re3=   c1
  re4="'"
  re5= "?a  ?pred2  ?b. ?b ?pred3  ?c."
  re6="?c  ?pred4     "
  re7="'"
  re8=c2
  re9="'"
  re10="}"
  res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10
  query_mesh=str(res_new)
  res_n = g1.query(query_mesh)
  path=""
  for row in res_n:
    if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["pred2"].rsplit('/', 1)[-1]
      prop3=row["pred3"].rsplit('/', 1)[-1]
      prop4=row["pred4"].rsplit('/', 1)[-1]

    path=str(prop1)+str(prop2)+str(prop3)+str(prop4)     
  return path


def three(c1,c2,rel):
  re1= " SELECT  ?pred1 ?pred2 ?pred3  ?pred4 ?pred5 WHERE  {?a    ?pred1    "
  re2="'"
  re3=   c1
  re4="'"
  re5= "?a  ?pred2  ?b. ?b ?pred3  ?c. ?d  ?pred4  ?e"
  re6="?e  ?pred5     "
  re7="'"
  re8=c2
  re9="'"
  re10="}"
  res_new=re1+re2+re3+re4+re5+re6+re7+re8+re9+re10
  query_mesh=str(res_new)
  path=""
  res_n = g1.query(query_mesh)
  for row in res_n:
    if(len(row)>0):
      prop1=row["pred1"].rsplit('/', 1)[-1]
      prop2=row["pred2"].rsplit('/', 1)[-1]
      prop3=row["pred3"].rsplit('/', 1)[-1]
      prop4=row["pred4"].rsplit('/', 1)[-1]
      prop5=row["pred5"].rsplit('/', 1)[-1]

    path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)
  return path

def axiom1(c1,c2,rel):
  
     q1="Select ?pred1 ?b  ?d  ?f  where{{?a  ?pred1  ?all1}{?a ?b  _:c} { _:c  ?d ?e} {?e  ?f  ?g }"
     q2="FILTER regex(?all1,'"
     q3=c1+"') FILTER regex(?g,'"+c2+"')}"
     query=q1+q2+q3
     print(query)
     result=g1.query(query)
     path=""
     for row in result:
         if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["b"].rsplit('/', 1)[-1]
          prop3=row["d"].rsplit('/', 1)[-1]
          prop4=row["f"].rsplit('/', 1)[-1]

         path=str(prop1)+str(prop2)+str(prop3)+str(prop4)
      
     return path


def axiom2(c1,c2,rel):
  
     q1="Select ?pred1 ?pred2 ?b ?d ?f   where{{?a  ?pred1  ?all0}{?a  ?pred2  ?all1}{?all1 ?b  _:c} { _:c  ?d ?e} {?e  ?f  ?g }"
     q2="FILTER regex(?all1,'"
     q3=c1+"') FILTER regex(?g,'"+c2+"')}"
     query=q1+q2+q3
     print(query)
     result=g1.query(query)
     path=""
     for row in result:
         if(len(row)>0):
          prop1=row["pred1"].rsplit('/', 1)[-1]
          prop2=row["pred2"].rsplit('/', 1)[-1]
          prop3=row["b"].rsplit('/', 1)[-1]
          prop4=row["d"].rsplit('/', 1)[-1]
          prop5=row["f"].rsplit('/', 1)[-1]

         path=str(prop1)+str(prop2)+str(prop3)+str(prop4)+str(prop5)
      
     return path
                
