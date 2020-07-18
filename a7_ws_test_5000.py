# =============================================================================
# TFM Fernando Ca�avate Vega - Master of AI - Polytechnic University of Madrid
# Script: Word similarity test
# =============================================================================



# =============================================================================
# evalua la similitud entre palabras tomando como referencia sinónimos del dissionario  Espasa Calpe
# =============================================================================




#from google.colab import drive
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import string
#drive.mount('/content/gdrive')
#root_dir = "/content/gdrive/My Drive/"
#def ls2(path): 
#    return [obj.name for obj in scandir(path) if obj.is_file()]



base_verb_pd=pd.read_csv("base_verb_5000.csv",header=0,names=['index_verb','verbostd','verbo'])
base_verb_pd.index = np.arange(1, len(base_verb_pd)+1)
base_noun_pd=pd.read_csv("base_noun_5000.csv",header=0,names=['index_noun','nombrestd'])
base_noun_pd.index = np.arange(1, len(base_noun_pd)+1)
coord_nounsuj_verb_pd=pd.read_csv("coord_nounsuj_verb_5000_densa.csv")
coord_nounsuj_verb_pd.index = np.arange(1, len(coord_nounsuj_verb_pd)+1)

# def addceroelement():
#     #coord_nounsuj_verb_pd=coord_nounsuj_verb_pd.append(coord_nounsuj_verb_pd.Series(0, index=df.columns), ignore_index=True)
#     coord_nounsuj_verb_pd.loc[len(coord_nounsuj_verb_pd)] = 0
#     base_noun_pd.loc[len(base_noun_pd)]=[499,500,"cero",0]
#     #print (coord_nounsuj_verb_pd.loc[len(coord_nounsuj_verb_pd)])
#     #print (base_noun_pd.loc[len(base_noun_pd)])


def indexnombre(nombre):
    if ((base_noun_pd['nombrestd']==nombre).any()):
        filanombre=base_noun_pd[(base_noun_pd['nombrestd']==nombre)]
        indexn=filanombre.iloc[0][0]
    else:
        indexn=0
    return(indexn)       
def indexverbo(verbo):
    filaverbo=base_verb_pd[(base_verb_pd['verbostd']==verbo)]
    #print (filaverbo.iloc[0][0])
    indexv=int(filaverbo.iloc[0][0])
    return(indexv)
    
def nombre(indicenombre):
    filanombre=base_noun_pd[(base_noun_pd['index_noun']==indicenombre)]
    nombre=filanombre.iloc[0][1]
    return(nombre)    
def verbo(indiceverbo):
    filaverbo=base_verb_pd[(base_verb_pd['index_verb']==indiceverbo)]
    verbo = filaverbo.iloc[0][1]
    return(verbo)   

#obtenemso el vector correspondiente a un nombre dado
def vector_noun_suj(nombre):
    index_noun=indexnombre(nombre)
    vectornombre=coord_nounsuj_verb_pd.iloc[index_noun,:]
    return(vectornombre)

def vector_noun_suj_index(index):
    index_noun=index
    #print (index_noun)
    vectornombre=coord_nounsuj_verb_pd.iloc[index_noun,:]
    return(vectornombre)
def vector_verb_index(index):
    print (index)
    print (coord_nounsuj_verb_pd.columns.values)
    headersrow=pd.DataFrame({'indexverb':coord_nounsuj_verb_pd.columns.values,'indexreal':np.arange(0, len(coord_nounsuj_verb_pd.columns.values))})
    indexcolumn=headersrow[(headersrow['indexverb']==str(index))]
    index_column_value=indexcolumn.iloc[0].loc['indexreal']
    print (index_column_value)
    npzeros=np.zeros(len(coord_nounsuj_verb_pd.columns))
    row_coord=pd.DataFrame(columns=coord_nounsuj_verb_pd.columns.values)
    row_coord.loc[0,:]=npzeros
    #print (row_coord)
    row_coord.iloc[0,index_column_value]=1
    return (row_coord)

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

def distanciasuj_index_vector(indexnombre1,vector):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(indexnombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray(vector))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    diferencia=vector1n-vector2n
    distancia=np.linalg.norm(diferencia)
    return([distancia,angulo])


def suma_index(indexnombre1,indexnombre2):
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
    vectorsuma=vector1n+vector2n
    modulovectorsuma=np.linalg.norm(vectorsuma)
    if modulovectorsuma==0:
        modulovectorsuma=1
    vectorsuma=vectorsuma/modulovectorsuma
    return(vectorsuma)

def resta_index(indexnombre1,indexnombre2):
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
    for i in range(0,len(vector1)):
        if vector1n[i]>0 and vector2n[i]>0:
            vector1n[i]=0
            vector2n[i]=0
        if vector1n[i]==0 and vector2n[i]>0:
            vector1n[i]=0
            vector2n[i]=0
    vectorresta=vector1n-vector2n
    vectorresta=np.absolute(vectorresta)
    modulovectorresta=np.linalg.norm(vectorresta)
    if modulovectorresta==0:
        modulovectorresta=1
    vectorresta=vectorresta/modulovectorresta
    return(vectorresta)

def resta_vector(vector1,vector2):
    vector1=np.squeeze(np.asarray((vector1)))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector2)))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    for i in range(0,len(vector1)):
        if vector1n[i]>0 and vector2n[i]>0:
            vector1n[i]=0
            vector2n[i]=0
        if vector1n[i]==0 and vector2n[i]>0:
            vector1n[i]=0
            vector2n[i]=0
    vectorresta=vector1n-vector2n
    vectorresta=np.absolute(vectorresta)
    modulovectorresta=np.linalg.norm(vectorresta)
    if modulovectorresta==0:
        modulovectorresta=1
    vectorresta=vectorresta/modulovectorresta
    return(vectorresta)

def suma_vector(vector1,vector2):
    vector1=np.squeeze(np.asarray((vector1)))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector2)))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    vectorsuma=vector1n+vector2n
    vectorsuma=np.absolute(vectorsuma)
    modulovectorsuma=np.linalg.norm(vectorsuma)
    if modulovectorsuma==0:
        modulovectorsuma=1
    vectorsuma=vectorsuma/modulovectorsuma
    vectorsuma_pd=pd.Series(vectorsuma)
    return(vectorsuma_pd)


#primeros 100 nombres mas cercanos al nombre dado
def buscar_nombre_cercano(argumentonombre):
    index_argumentonombre=indexnombre(argumentonombre)
    matrizdistancias=[]
    for i_nombre in range (1,498):
        distancia=distanciasuj_index (index_argumentonombre,i_nombre)
        matrizdistancias.append([nombre(i_nombre),distancia[0],])
    matrizdistancias_pd=pd.DataFrame(matrizdistancias,columns=['nombres','distancia'])
    nombres = matrizdistancias_pd.sort_values('distancia')   
    return (nombres[:20])

def buscar_nombre_cercano_a_vector(vector):
    matrizdistancias=[]
    for i_nombre in range (1,498):
        distancia=distanciasuj_index_vector(i_nombre,vector)
        matrizdistancias.append([nombre(i_nombre),distancia[0],])
    matrizdistancias_pd=pd.DataFrame(matrizdistancias,columns=['nombres','distancia'])
    nombres = matrizdistancias_pd.sort_values('distancia')   
    return(nombres[1:20])

def suma_nombres(nombre1,nombre2):
    indexnombre1=indexnombre(nombre1)
    indexnombre2=indexnombre(nombre2)
    vectorsuma=suma_index(indexnombre1,indexnombre2)
    nombres=buscar_nombre_cercano_a_vector(vectorsuma)
    #print (nombres[1:20])
    return (nombres)

def suma_nombres_verbos(nombre,verbo):
    indexnombre1=indexnombre(nombre)
    indexverbo1=indexverbo(verbo)
    vectornombre=vector_noun_suj_index(indexnombre1)
    vectorverbo=vector_verb_index(indexverbo1)
    print ("indexverbo",indexverbo1)
    vectorsuma=suma_vector(vectornombre,vectorverbo)
    nombres=buscar_nombre_cercano_a_vector(vectorsuma)
    listadoverbos(vectorsuma)
    #print (nombres[1:20])
    return (nombres)

def resta_nombres(nombre1,nombre2):
    indexnombre1=indexnombre(nombre1)
    indexnombre2=indexnombre(nombre2)
    vectorresta=resta_index(indexnombre1,indexnombre2)
    nombres=buscar_nombre_cercano_a_vector(vectorresta)
    #print (nombres[1:20])
    return (nombres)

def listadoverbos(vector):
    vector_matrix=vector.values
    print (vector_matrix)
    listado_verbs_pd=pd.DataFrame({"coord":vector_matrix,"verbos":base_verb_pd['verbostd']})
    listado_ordenado_pd=listado_verbs_pd.sort_values('coord', ascending=False)
    return (listado_ordenado_pd.iloc[0:30,:])
    


def resultado_listado_verbos(vector):
    verbo=listadoverbos(vector)
    base_noun_verb=[]
    for index,row in verbo.iterrows():
        print (row.iloc[0],row.iloc[1])
        base_noun_verb.append([row.iloc[0],row.iloc[1]])
    listado_verbos_pd=pd.DataFrame(base_noun_verb,columns=["weight","verb"])
    #listado_verbos_pd.to_csv("listado_verbos.csv",index=False,encoding="utf_8_sig")

def nombres_diferencia(nombre1,nombre2):
    vector_nombre1=vector_noun_suj(nombre1)
    vector_nombre2=vector_noun_suj(nombre2)
    diferenciavect=resta_vector(vector_nombre1,vector_nombre2)
    diferenciavect_pd=pd.Series(diferenciavect)
    #diferenciavect_pd=pd.DataFrame.from_records(diferenciavect.reshape(-1).tolist())
    resultado_listado_verbos(diferenciavect_pd)
    print(buscar_nombre_cercano_a_vector(diferenciavect_pd))
    return (diferenciavect_pd)    
 
def nombres_suma(nombre1,nombre2):
    vector_nombre1=vector_noun_suj(nombre1)
    vector_nombre2=vector_noun_suj(nombre2)
    sumavect=suma_vector(vector_nombre1,vector_nombre2)
    sumavect_pd=pd.Series(sumavect)
    resultado_listado_verbos(sumavect_pd)
    print(buscar_nombre_cercano_a_vector(sumavect_pd))
    return (sumavect_pd) 
   


def sinonimos_esp(enlace):
    #print ("enlace",enlace)
    sinonimos=[]
    url='http://www.wordreference.com/sinonimos/'
    buscar=url+enlace
    resp=requests.get(buscar)
    bs=BeautifulSoup(resp.text,'lxml')
    lista=bs.find_all(class_='trans clickable')
    if lista:
        for sin in lista:
            sino=sin.find_all('li')
            fin=sino[0]
            #print ("fin.next_element",fin.next_element)
            if fin:
                sinonimos=fin.next_element.split(', ')           
                return(sinonimos)
 

base_noun_distancias=[]

for rownombre in range (1,4990):   
    nombre_base=base_noun_pd.iloc[rownombre][1]
    print ("nombre_base",nombre_base)
    index_palabra1=indexnombre(nombre_base)
    sinonimos_matriz=sinonimos_esp(nombre_base)
    if sinonimos_matriz:
        for sinonimo in sinonimos_matriz:
            print("sinonimo:",sinonimo)
            index_palabra2=indexnombre(sinonimo)
            if index_palabra1>0 and index_palabra2>0:
                distancia=distanciasuj_index(index_palabra1,index_palabra2)
                print("distancia=",distancia)
                base_noun_distancias.append([index_palabra1,nombre_base,index_palabra2,sinonimo,distancia[0],distancia[1]])
base_noun_distancias_pd=pd.DataFrame(base_noun_distancias)
#base_noun_distancias_pd=pd.DataFrame(base_noun_distancias,columns:["index_base","nombre_base","index_sinónimo","sinonimo","distancia_eucl","distancia_cos"])
base_noun_distancias_pd.to_csv("base_noun_pd_distancias_5000.csv",encoding="utf_8_sig")

