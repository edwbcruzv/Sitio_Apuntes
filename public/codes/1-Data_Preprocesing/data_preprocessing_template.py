# -*- coding: utf-8 -*-

# =============================================================================
# --------------------Plantilla de Pre-procesado--------------------
# =============================================================================

# =============================================================================
# --------------------Importando librerias--------------------
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

dataset = pd.read_csv('Data.csv') # {buscar el dataset}

# Variables independiente:Mayuscula por ser una matriz.
#   tomamos [Todas las filas ,Solo la columna(s)...]
X = dataset.iloc[:,0:3].values # {se pueden modificar segun se necesite}

# Variable dependiente:minuscula por ser un vector.
#   tomamos [Todas las filas: Solo la ultima columna]
y = dataset.iloc[:,-1].values # {se pueden modificar segun se necesite}

# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================

from sklearn.impute import SimpleImputer
# Los valores desconocidos de los valores independientes son los NA´s.
# El valor que se va a sustituir que sera la media.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Seleccionando las columnas las cuales estan los valores NA´s.
imputer.fit(X[:,1:3]) # ajustando valores
# Sobreescribirnedo la matriz con la nueva trasformacion configurada.
X[:,1:3] = imputer.transform(X[:,1:3])


# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# la funcion se encargara de transformar las categorias a datos numericos.
le_X =LabelEncoder()
# De la tabla de variables independientes se toma 
# la columna "Country"y todas las filas. Y se sobreescribe la tabla.
X[:,0]=le_X.fit_transform(X[:,0])

# variables dummy, creando una columna por cada categoria y de las variables 
# dummy solo se marca la categoria correcta con un booleano.


ct = ColumnTransformer(
    # Lista de tuplas (nombre,transformador,columnas) que se le aplicara 
    # al conjunto de datos.
    [('one_hot_encoder',OneHotEncoder(categories='auto',dtype=int),[0])],
    # Se pasa el resto de columnas que no se tocaron.
    remainder='passthrough')
# # Se creara una columna por cada pais (categorizando)
# # Y con un booleano se sabra el pais correspondiente (onehotencoder)

X = np.array(ct.fit_transform(X),dtype=float)

#  
le_y=LabelEncoder()
y=le_y.fit_transform(y)

# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings------------------
# =============================================================================

from sklearn.model_selection import train_test_split
# la sig funcion devolvera varias variables con los valores de testing y training

# Como parametros:Matriz independiente,
#       matriz dependiente a predecir,
#       tamaño del conjunto de testing en % (el resto se va a entrenamiento),
#       numero random de division de datos (semilla random=cualquier numero).
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Escalador para las variables independientes
sc_X = StandardScaler()
# Escalando variables de training, se usa el fit_trasform
X_train = sc_X.fit_transform(X_train)
# Se escala con el mismo escalador con las variables de testing con transform
# para que la trasformacion lo haga en base al conjunto escalado de training
X_test = sc_X.transform(X_test)



