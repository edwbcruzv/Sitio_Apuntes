# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:38:05 2023

@author: edwin
"""

# =============================================================================
# Regresion Lineal Multiple
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

dataset = pd.read_csv('50_Startups.csv')

# Variable independiente:Mayuscula por ser una matriz.
X = dataset.iloc[:,:-1].values

# Variable dependiente:minuscula por ser un vector.
y = dataset.iloc[:,4].values

# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# la funcion se encargara de transformar las categorias a datos numericos.
le_X =LabelEncoder()
# De la tabla de variables independientes se toma 
# la columna "State"y todas las filas. Y se sobreescribe la tabla.
X[:,3]=le_X.fit_transform(X[:,3])

# Ahora se debe de transformar la columna "State" a variables dummy,
# creando una columna por cada categoria y de las variables dummy solo se
# marca la categoria correcta con un booleano.

ct = ColumnTransformer(
    # Lista de tuplas (nombre,transformador,columnas) que se le aplicara 
    # al conjunto de datos.
    [('one_hot_encoder',OneHotEncoder(categories='auto',dtype=int),[3])],
    # Se pasa el resto de columnas que no se tocaron.
    remainder='passthrough')

# Se creara una columna por cada estado (categorizando)
# Y con un booleano se sabra el estado correspondiente (onehotencoder)
# Se van a cambiar de posision: pasaran al principio del dataset en X.
X=ct.fit_transform(X)
# Se debe de convertir a array, ya que trabajamos con ndarrays 
X = np.array(X,dtype=float)

# Para no caer en la trampa de las variables dummy, eliminaremos la 
# primera columna y conservaremos el resto:
# (si en ambos dummys es 0, entonces es la categoria eliminada)
X=X[:,1:]


# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings--------------------
# =============================================================================

from sklearn.model_selection import train_test_split
# la sig funcion devolvera varias variables con los valores de testing y training
# Como parametros:Matriz independiente,
#           matridependiente a predecir,
#           tamaño del conjunto de testing en % (el resto se va a entrenamiento),
#           numero random de division de datos (semilla random=cualquier numero).
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================

# no se necesita escalado

# =============================================================================
# --------------------Ajustando el modelo de Regresion Lineal Multiple---------
# --------------------con el conjunto de entrenamiento--------------------
#
# Usando todas (3+ State:3 son dummys) las variables independientes
# =============================================================================

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# prediccion de los resultados en el conjunto de testing
y_pred=regression.predict(X_test)

# =============================================================================
# -----------Ajustando el modelo optimo de Regresion Lineal Multiple-----------
# 
# Eliminacion hacia atras
# Usando todas (3+ State:3 son dummys) las variables independientes
# =============================================================================

import statsmodels.api as sm

# Agregamos una columna de unos al principio de la matriz,
# el cual tomara el lugar el termino indepeniente para calcular el p-valor.
# axis = 1 es para agregarlo en columna.
# Se crea el arreglo de 1's y al final se agregan las columnas de la matriz X.
X=np.append(arr=np.ones((50,1),dtype=int),values=X,axis=1) 

# =============================================================================
# PASO 1: SELECCIONAR EL NIVEL DE SIGNIFICACION
SL=0.05
# =============================================================================

# =============================================================================
# PASO 2: SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORAS
# matriz de caracteristicas optimas
# variables significativas (empezamos con todas las variables)
X_opt=X[:,[0,1,2,3,4,5]]
regression_OLS=sm.OLS(y,X_opt).fit()
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
print(regression_OLS.summary())
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que x3 > SL, entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA
X_opt=X[:,[0,1,2,4,5]] # se elimino x3
# =============================================================================

# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression_OLS=sm.OLS(y,X_opt).fit()
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
print(regression_OLS.summary())
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que x2 > SL, entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA
X_opt=X[:,[0,1,4,5]] # se elimino x2
# =============================================================================

# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression_OLS=sm.OLS(y,X_opt).fit()
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
print(regression_OLS.summary())
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que x3 > SL, entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA
X_opt=X[:,[0,1,4]] # se elimino x3
# =============================================================================

# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression_OLS=sm.OLS(y,X_opt).fit()
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
print(regression_OLS.summary())
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que x2 > SL,pero se puede llegar hasta este punto,
# ya que si eliminamos una variable mas llegariamos a una regresion lineal simple. 
# SINO AL FIN.
# =============================================================================





