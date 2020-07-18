

# =============================================================================
# realiza ciertas operaciones entre vectores sobre matriz densa de 500 palabras
# formato de las funciones operaciones [resultado][tiporesultado]_[argumentos]
#=============================================================================

import numpy as np
import pandas as pd

base_verb_pd=pd.read_csv("base_verb_5000.csv",header=0,names=['index_verb','verbostd','verbo'])
base_verb_pd.index = np.arange(1, len(base_verb_pd)+1)
base_noun_pd=pd.read_csv("base_noun_5000.csv",header=0,names=['index_noun','nombrestd'])
base_noun_pd.index = np.arange(1, len(base_noun_pd)+1)
coord_nounsuj_verb_pd=pd.read_csv("coord_nounsuj_verb_5000_densa.csv")
coord_nounsuj_verb_pd.index = np.arange(1, len(coord_nounsuj_verb_pd)+1)

def index_nombre(nombre):
    filanombre=base_noun_pd[(base_noun_pd['nombrestd']==nombre)]
    indexn=int(filanombre.iloc[0][0])
    return(indexn)    
def index_verbo(verbo):
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
    index_noun=index_nombre(nombre)
    vectornombre=coord_nounsuj_verb_pd.iloc[index_noun,:]
    return(vectornombre)

def vector_noun_suj_index(index):
    index_noun=index
    #print (index_noun)
    vectornombre=coord_nounsuj_verb_pd.iloc[index_noun,:]
    return(vectornombre)
def vector_verb_index(index):
    #print (index)
    #print (coord_nounsuj_verb_pd.columns.values)
    headersrow=pd.DataFrame({'indexverb':coord_nounsuj_verb_pd.columns.values,'indexreal':np.arange(0, len(coord_nounsuj_verb_pd.columns.values))})
    indexcolumn=headersrow[(headersrow['indexverb']==str(index))]
    index_column_value=indexcolumn.iloc[0].loc['indexreal']
    #print (index_column_value)
    npzeros=np.zeros(len(coord_nounsuj_verb_pd.columns))
    row_coord=pd.DataFrame(columns=coord_nounsuj_verb_pd.columns.values)
    row_coord.loc[0,:]=npzeros
    #print (row_coord)
    row_coord.iloc[0,index_column_value]=1
    return (row_coord)

def noceros_index(index_nombre3):
    vectornombre=vector_noun_suj_index(index_nombre3)
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

def distanciasuj_index(index_nombre1,index_nombre2):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre2))))
    modulovector2=np.linalg.norm(vector2)
    if modulovector2==0:
        modulovector2=1
    vector2n=vector2/modulovector2
    angulo=np.dot(vector1n,vector2n)
    diferencia=vector1n-vector2n
    distancia=np.linalg.norm(diferencia)
    return([distancia,angulo])

def distanciasuj_index_vector(index_nombre1,vector):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre1))))
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

##### OPERACIONES #########
def sumavector_index(index_nombre1,index_nombre2):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre2))))
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

def restavector_index(index_nombre1,index_nombre2):
    vector1=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre1))))
    modulovector1=np.linalg.norm(vector1)
    if modulovector1==0:
        modulovector1=1
    vector1n=vector1/modulovector1
    vector2=np.squeeze(np.asarray((vector_noun_suj_index(index_nombre2))))
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

def restavector_vector(vector1,vector2):
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

def diferenciavector_vector(vector1,vector2):
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
    vectorresta=vector1n-vector2n
    vectorresta=np.absolute(vectorresta)
    modulovectorresta=np.linalg.norm(vectorresta)
    if modulovectorresta==0:
        modulovectorresta=1
    vectorresta=vectorresta/modulovectorresta
    return(vectorresta)

def sumavector_vector(vector1,vector2):
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

def productovector_vector(vector1,vector2):
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
    vector_producto=np.zeros(len(vector1))
    for i in range(0,len(vector1)):
        vector_producto[i]=vector1n[i]*vector2n[i]
    modulovector_producto=np.linalg.norm(vector_producto)
    if modulovector_producto==0:
        modulovector_producto=1
    vector_producto=vector_producto/modulovector_producto
    vector_producto_pd=pd.Series(vector_producto)
    return(vector_producto_pd)

#primeros 100 nombres mas cercanos al nombre dado
def buscarnombrecercano_nombre(argumentonombre):
    index_argumentonombre=index_nombre(argumentonombre)
    matrizdistancias=[]
    for i_nombre in range (1,4995):
        distancia=distanciasuj_index (index_argumentonombre,i_nombre)
        matrizdistancias.append([nombre(i_nombre),distancia[0],])
    matrizdistancias_pd=pd.DataFrame(matrizdistancias,columns=['nombres','distancia'])
    nombres = matrizdistancias_pd.sort_values('distancia')   
    return (nombres[:20])

def buscarnombrecercano_vector(vector):
    matrizdistancias=[]
    for i_nombre in range (1,4995):
        distancia=distanciasuj_index_vector(i_nombre,vector)
        matrizdistancias.append([nombre(i_nombre),distancia[0],])
    matrizdistancias_pd=pd.DataFrame(matrizdistancias,columns=['nombres','distancia'])
    nombres = matrizdistancias_pd.sort_values('distancia')   
    return(nombres[1:20])

def sumanombres_nombres(nombre1,nombre2):
    index_nombre1=index_nombre(nombre1)
    index_nombre2=index_nombre(nombre2)
    vectorsuma=sumavector_index(index_nombre1,index_nombre2)
    nombres=buscarnombrecercano_vector(vectorsuma)
    #print (nombres[1:20])
    return (nombres)

def sumanombres_nombresverbos(nombre,verbo):
    index_nombre1=index_nombre(nombre)
    index_verbo1=index_verbo(verbo)
    vectornombre=vector_noun_suj_index(index_nombre1)
    vectorverbo=vector_verb_index(index_verbo1)
    #print ("index_verbo",index_verbo1)
    vectorsuma=sumavector_vector(vectornombre,vectorverbo)
    nombres=buscarnombrecercano_vector(vectorsuma)
    listadoverbos_vector(vectorsuma)
    #print (nombres[1:20])
    return (nombres)

def listadoverbos_vector(vector):
    vector_matrix=vector.values
    #print (vector_matrix)
    listado_verbs_pd=pd.DataFrame({"coord":vector_matrix,"verbos":base_verb_pd['verbostd']})
    listado_ordenado_pd=listado_verbs_pd.sort_values('coord', ascending=False)
    return (listado_ordenado_pd.iloc[0:30,:])

def restavector_nombres(nombre1,nombre2):
    vector_nombre1=vector_noun_suj(nombre1)
    vector_nombre2=vector_noun_suj(nombre2)
    restavect=restavector_vector(vector_nombre1,vector_nombre2)
    restavect_pd=pd.Series(restavect)
    return (restavect_pd)    
 
def diferenciavector_nombres(nombre1,nombre2):
    vector_nombre1=vector_noun_suj(nombre1)
    vector_nombre2=vector_noun_suj(nombre2)
    diferenciavect=diferenciavector_vector(vector_nombre1,vector_nombre2)
    diferenciavect_pd=pd.Series(diferenciavect)
    return (diferenciavect_pd)    

def sumavector_nombres(nombre1,nombre2):
    vector_nombre1=vector_noun_suj(nombre1)
    vector_nombre2=vector_noun_suj(nombre2)
    sumavect=sumavector_vector(vector_nombre1,vector_nombre2)
    sumavect_pd=pd.Series(sumavect)
    listadoverbos_vector(sumavect_pd)
    return (sumavect_pd) 

def productovector_nombres(nombre1,nombre2):
    vectornombre1=vector_noun_suj(nombre1)
    vectornombre2=vector_noun_suj(nombre2)
    productovector=productovector_vector(vectornombre1,vectornombre2)
    return(productovector)   

    
   
#print(vector_noun_suj("canción"))



### TEST busqueda de nombres cercanos
#resultado=buscarnombrecercano_vector(vector_noun_suj("alcalde"))
 
### TEST sociología diferencias de sexo
#resultado=hombre_menos_mujer=restavector_nombres("hombre","mujer")
#mujer_menos_hombre=restavector_nombres("mujer","hombre")


### TEST rey-hombre+mujer=reina
# rey_menos_hombre=restavector_nombres("rey","hombre")   
# rey_menos_hombre_mas_mujer=sumavector_vector(rey_menos_hombre,vector_noun_suj("mujer"))
# print(listadoverbos_vector(rey_menos_hombre_mas_mujer))
# resultado=buscarnombrecercano_vector(rey_menos_hombre_mas_mujer)

### TEST composición de nombres y vectores
#print ("vehiculo+volar=",sumanombres_nombresverbos("vehículo","volar"))
#print ("vehiculo+navegar=",sumanombres_nombresverbos("vehículo","navegar"))

### TEST diferencia de nombres
#print (restavector_nombres("barco","vehículo"))
#print (restavector_nombres("avión","vehículo"))

### TEST abstraccion
barco_por_avion=productovector_nombres("barco","avión")
barco_por_avion_menos_casa=restavector_vector(barco_por_avion,vector_noun_suj("casa"))
print (listadoverbos_vector(barco_por_avion_menos_casa))
resultado=buscarnombrecercano_vector(barco_por_avion_menos_casa)

#### TEST nombres cercanos a un verbo
#resultado=buscarnombrecercano_vector(vector_verb_index(index_verbo("estudiar")))

### TEST analisis de diferencias entre palabras parecidas
#resultadoproducto=listadoverbos_vector(productovector_nombres("principio","origen"))
#print ("resultadoproducto",resultadoproducto)
#resultadodiferencia=listadoverbos_vector(diferenciavector_nombres("principio","origen"))
#print ("resultadodiferencia",resultadodiferencia)
#resultadoresta_principio_origen=listadoverbos_vector(restavector_nombres("principio","origen"))
#print ("resultadoresta_principio-origen",resultadoresta_principio_origen)
#resultadoresta_origen_principio=listadoverbos_vector(restavector_nombres("origen","principio"))
#print ("resultadoresta_origen-principio",resultadoresta_origen_principio)


### IMPRESION Y SALVADO DE RESULTADOS
print (resultado)
#resultado.to_csv("resultado.csv",encoding="utf_8_sig")