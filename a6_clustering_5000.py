# =============================================================================
# TFM Fernando Cañavate Vega - Master of AI - Polytechnic University of Madrid
# Script: name vectors clustering
# =============================================================================



# =============================================================================
# realiza clusterizacion de los nombres utilizando sus vectores con kmeans, y drendograma
# =============================================================================




import rdflib
import openpyxl
from os import listdir
from os import scandir
import xlrd
#from google.colab import drive
from openpyxl import load_workbook
from openpyxl import Workbook
from urllib import request
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy import stats
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

#from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds
#https://pypi.org/project/sparsesvd/
#drive.mount('/content/gdrive')
#root_dir = "/content/gdrive/My Drive/"
#def ls2(path): 
#    return [obj.name for obj in scandir(path) if obj.is_file()]


base_verb_pd=pd.read_csv("base_verb_5000.csv",header=0,names=['index_verb','verbostd','verbo'])
base_noun_pd=pd.read_csv("base_noun_5000.csv",header=0,names=['index_noun','nombrestd'])
nombres_vectores_pd=base_noun_pd
vectores_pd=pd.read_csv("coord_nounsuj_verb_5000_densa.csv")
                            
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
    nombre=filanombre.iloc[0][1]
    return(nombre)  

def verbo(indiceverbo):
    filaverbo=base_verb_pd[(base_verb_pd['index_verb']==indiceverbo)]
    verbo = filaverbo.iloc[0][1]
    return(verbo)   



def vector_noun_suj_index(index):
    index_noun=index
    vectornombre=coord_nounsuj_verb_matrix.getrow(index_noun)
    vector_noun_dense=vectornombre.todense()
    return(vector_noun_dense)
def vector_noun_suj(nombre):
    index_noun=indexnombre(nombre)
    vectornombre=coord_nounsuj_verb_matrix.getrow(index_noun)
    vector_noun_dense=vectornombre.todense()
    return(vector_noun_dense)

def listadoverbos(vector1):
    vector=vector1.transpose()
    vector.reset_index(drop=True,inplace=True)
    listado_verbs_pd=pd.concat([vector,base_verb_pd['verbostd']],axis=1,join="inner")
    listado_verbs_pd.columns=["coord","verbos"]
    listado_verbs_pd_ordenado=listado_verbs_pd.sort_values(by=['coord'],ascending=False)
    #print (listado_verbs_pd.shape)
    #print (listado_verbs_pd)
    print (listado_verbs_pd_ordenado.iloc[0:10,:])
    #return (listado_verbs_pd_ordenado.iloc[0:10,:].values.tolist())
    #print (listado_verbs_pd_ordenado.iloc[0:10,:])
    return (listado_verbs_pd_ordenado.iloc[0:10,:])
def listadoverbos_nombre(nombre1):
    #vector (primeros 100 elementos a un nombre dado)
    vec=vector_noun_suj(nombre1)
    vec_pd=pd.DataFrame(vec)
    tabla_verbos=[]
    for a in range(1,53094):
        tabla_verbos.append([vec_pd.iloc[0,a],verbo(a)])
    listado_verbs_pd=pd.DataFrame(tabla_verbos)
    listado_verbs_pd.columns=["coord","verbos"]
    listado_verbs_pd_ordenado=listado_verbs_pd.sort_values(by=['coord'],ascending=False)
    return (listado_verbs_pd_ordenado.iloc[0:20,:])
def listadoverbos_cluster(vector):
    listado_verbs_pd=pd.DataFrame({"coord":vector,"verbos":base_verb_500['verbostd']})
    listado_verbs_pd_ordenado=listado_verbs_pd.sort_values(by=['coord'],ascending=False)
    #print (listado_verbs_pd.shape)
    #print (listado_verbs_pd)
    #print (listado_verbs_pd_ordenado.iloc[0:10,:])
    return (listado_verbs_pd_ordenado.iloc[0:10,:])
def listado_a_cadenas (listadoverbos1):
    cadena="="
    for index,linea in listadoverbos1.iterrows():
        print("linea",linea)
        if type(linea['verbos']) == str and type(linea['coord'])==int:
            cadena=cadena+str(linea['coord'])+"x"+linea['verbos']+" + "
    print ("cadena",cadena)
    return cadena


#clusterizamos con Kmeans
kmeans=KMeans(n_clusters=500)
kmeans.fit(vectores_pd)
labels = kmeans.predict(vectores_pd)
centroids = kmeans.cluster_centers_
nombres_vectores_pd['clases']=labels
nombres_vectores_pd.to_csv("base_noun_5000_clusters.csv",index=False,encoding="utf_8_sig")
#print(labels)
#print(centroids)
centroids_pd=pd.DataFrame(centroids[:,0])
centroids_pd['verbos1'] = None
nombres_vectores_pd['verbos1']=None
centroids_aux=[]
nombres_vect_aux=[]

#for index, row in nombres_vectores_pd.iterrows():
#    print("index",index)
#    print ("row",row)
#    nombre1=row['word']
#    print ("word", nombre1)
#    nombres_vectores_pd.loc[index,'verbos1']=listado_a_cadenas(listadoverbos_nombre(nombre1))
#    if index%10:
#        nombres_vectores_pd.to_csv("nombres_vectores_500_3_labels_verbos_1.csv",encoding="utf_8_sig")
#nombres_vectores_pd.to_csv("nombres_vectores_500_3_labels_verbos.csv_1",encoding="utf_8_sig")
#Clustering jerárquico
# Creamos el dendograma para encontrar el número óptimo de clusters
label_palabras=nombres_vectores_pd['nombrestd']
#print (vectores_pd)
R=dendrogram = sch.dendrogram(sch.linkage(vectores_pd, metric='cosine', method = 'single'),p=100,truncate_mode="lastp",orientation='top')
temp = {R["leaves"][ii]: labels[ii] for ii in range(len(R["leaves"]))}
def llf(xx):
    return (label_palabras.iloc[temp[xx]])
dendrogram = sch.dendrogram(sch.linkage(vectores_pd, metric='cosine',method = 'single'),p=100,leaf_label_func=llf,truncate_mode="lastp",orientation='top')
plt.title('Dendrograma')
plt.xlabel('Distancias')
plt.ylabel('Palabras')
plt.show()


