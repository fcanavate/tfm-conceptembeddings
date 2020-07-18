# =============================================================================
# TFM Fernando Ca�avate Vega - Master of AI - Polytechnic University of Madrid
# Script: Samples generator of 500 elements with more representation
# =============================================================================

# =============================================================================
# obtiene una lista reducida a los 500 nombres mas representados, con sus
# coordenadas correspondientes, y la base de verbos de la que dependen. Elimina la columnas
# nulas y obtiene los primeros 20 verbos y su peso en cada uno los 500 elementos 
# la base
# realiza tambien una clusterizacion previa para comprar la representacio de embedding projector
# =============================================================================


import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
#cargamos las bases de nombre, verbos y coordenadas del corpus completo (fusioó)
base_verb_pd=pd.read_excel("base_verb_fusion.xlsx",names=['index_verb','verbostd','verbo'])
#base_verb_pd.index = np.arange(1, len(base_verb_pd)+1)
base_noun_pd=pd.read_excel("base_noun_fusion.xlsx",names=['index_noun','nombrestd','nombre'])
base_noun_pd.index = np.arange(1, len(base_noun_pd)+1)
coord_nounsuj_verb_pd=pd.read_excel("coord_nounsuj_verb_fusion.xlsx",names=['noun_suj_index','verbindex','coordenada'])
coord_nounsuj_verb_pd.index = np.arange(1, len(coord_nounsuj_verb_pd)+1)
#construimos matriz COO a partir de coord_nounsuj_verb_pd
coord_nounsuj_verb_matrix=coo_matrix((coord_nounsuj_verb_pd.loc[:,'coordenada'],(coord_nounsuj_verb_pd.loc[:,'noun_suj_index'],coord_nounsuj_verb_pd.loc[:,'verbindex'])),dtype=int)                                     
coord_nounsuj_verb_matrix.sum_duplicates() 
#añadirmos ellemento neutro a la base de verbos y nombres
base_verb_pd.iloc[0]=[0,"verbocero","verbocero"]
#base_noun_pd.iloc[0]=[0,"nombrecero","nombrecero"]
print ("inicio programa")                         
def indexnombre(nombre):
    filanombre=base_noun_pd[(base_noun_pd['nombrestd']==nombre)]
    indexn=filanombre.iloc[0][0]
    return(indexn)    
def indexverbo(verbo):
    filaverbo=base_verb_pd[(base_verb_pd['verbostd']==verbo)]
    indexv=filaverbo.iloc[0][0]
    return(indexv)
    
def nombre(indicenombre):
    filanombre=base_noun_pd[(base_noun_pd['index_noun']==indicenombre)]
    print (filanombre)
    print (filanombre.shape)
    nombre=filanombre.iloc[0][1]
    return(nombre)    
        
#obtenemso el vector correspondiente a un nombre dado
def vector_noun_suj(nombre):
    index_noun=indexnombre(nombre)
    vectornombre=coord_nounsuj_verb_matrix.getrow(index_noun)
    vector_noun_dense=vectornombre.todense()
    return(vector_noun_dense)

def vector_noun_suj_index(index):
    index_noun=index
    vectornombre=coord_nounsuj_verb_matrix.getrow(index_noun)
    vector_noun_dense=vectornombre.todense()
    return(vector_noun_dense)

def noceros(indexnombre3):
    vectornombre=vector_noun_suj_index(indexnombre3)
    noceros=np.count_nonzero(vectornombre)
    return (noceros)
def distanciasuj(nombre1,nombre2):
    vector1=np.squeeze(np.asarray((vector_noun_suj(nombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector_noun_suj(nombre2))))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    diferencia=vector1n-vector2n
    distancia=np.linalg.norm(diferencia)
    return([distancia,angulo])

def distanciasuj_index(indexnombre1,indexnombre2):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(indexnombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector_noun_suj_index(indexnombre2))))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    diferencia=vector1n-vector2n
    distancia=np.linalg.norm(diferencia)
    return([distancia,angulo])
def distancias(nombre1,nombre2):
    distanciasuj1=distanciasuj(nombre1,nombre2)
    print ("suj:",distanciasuj1[0])
def listadoverbos(vector):
    listado_verbs_pd=pd.DataFrame({"coord":vector,"verbos":base_verb_500['verbosstd']})
    listado_verbs_pd.sort_values('coord', ascending=False)
    print (listado_verbs_pd[0:10,:])
    return (listado_verbs_pd[0:10,:])

# Seleccionamos nombres en función de la cantidad de verbos que conjuguen en el corpus (nozeros)
matriznombres=[]
for indexnombre in range (1,10000):
    noceros1=noceros(indexnombre)
    matriznombres.append([indexnombre,nombre(indexnombre),noceros1])
matriznombres_pd=pd.DataFrame(matriznombres,columns=['index','nombres','noceros'])
nombresnozeros = matriznombres_pd.sort_values('noceros',ascending=False)
headers_vectores=base_verb_pd["index_verb"]
headers_vectores_matrix=headers_vectores.transpose().to_numpy()
vectores=pd.DataFrame (columns=headers_vectores_matrix)
nombres_vectores=[]
# elegimos las primeras 500 palabras dentro de la lista de nozeros que estará formada por elementos con el mayor número de coordnadas respecto a la bse no ceros
# la lista de los primeros 500 nombres la llamamos base_noun_500.csv
for nom1 in range (1,5000):
    indice_nombre=(nombresnozeros.iloc[nom1]['index'])
    #nombres_vectores.append([indice_nombre,nombre(indice_nombre),nom1])
    nombres_vectores.append([nom1-1,nombre(indice_nombre)])
    vector1=pd.DataFrame(vector_noun_suj_index(nombresnozeros.iloc[nom1]['index']))
    #vector2= (df - df.mean()) / (df.max() - df.min())
    vector2=np.squeeze(np.asarray(vector1))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    vector2=pd.DataFrame(vector2n)
    #print (vector2)
    vectores=pd.concat([vectores, vector2.transpose()], axis=0)
    
nombres_vectores_pd=pd.DataFrame(nombres_vectores,columns=['index_noun','nombrestd'])
#print (vectores)
#print (nombres_vectores)
#vectores.to_csv("vectores_completo_500.csv")
#nombres_vectores_pd.to_csv("nombres_vectores_500.csv",encoding="utf_8_sig")
#seleccion de columnas
vector_columna_nozeros=(vectores != 0).any(axis=0)
#print (vector_columna_nozeros)
#bas-verb_pd=base_verb_500=base_verb_pd[pd.concat([pd.DataFrame([True]), vector_columna_nozeros.transpose()], axis=0)]
base_verb_pd=base_verb_pd.drop(len(base_verb_pd)-1,axis=0)
#print(base_verb_pd)
base_verb_500=base_verb_pd[vector_columna_nozeros]
base_verb_500.to_csv("base_verb_5000.csv",index=False)
headers_vectores=base_verb_500["index_verb"]
headers_vectores_matrix=headers_vectores.transpose().to_numpy()
vectores=vectores.loc[:, headers_vectores_matrix]
vectores.to_csv("coord_nounsuj_verb_5000_densa.csv",index=False)
nombres_vectores_pd.to_csv("base_noun_5000.csv",index=False,encoding="utf_8_sig")


