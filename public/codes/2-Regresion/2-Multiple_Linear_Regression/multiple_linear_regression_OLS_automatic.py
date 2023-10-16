# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:47:35 2023

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

# Estructura de los datos:
# Contiene informacion de 50 empresas y sus gastos en 3 areas, lugar 
# donde se ubican y la ganancia 
# Filas :50
# Columnas:
#           |RyD Spend|Administration|Marketing Spend|State (vars independiente)
#           |Profit (var_dependiente)

dataset = pd.read_csv('50_Startups.csv')

# Variable independiente:Mayuscula por ser una matriz.
#   tomamos [Todas las filas , Todas las columnas, menos la ultima]
X = dataset.iloc[:,:-1].values

# Variable dependiente:minuscula por ser un vector.
#   tomamos [Todas las filas: Solo la ultima columna]
y = dataset.iloc[:,4].values

# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# Los datos son categorias que se deben de tranformar a numeros para
# que python los pueda trabajar.
# En este caso la columna "State".
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
# --Ajustando sutomaticamente el modelo optimo de Regresion Lineal Multiple----
# 
# Eliminacion hacia atras
# Usando todas (3+ State:3 son dummys) las variables independientes
# =============================================================================

import statsmodels.api as sm
import copy

# Agregamos una columna de unos al principio de la matriz,
# el cual tomara el lugar el termino indepeniente para calcular el p-valor.
# axis = 1 es para agregarlo en columna.
# Se crea el arreglo de 1's y al final se agregan las columnas de la matriz X.
X=np.append(arr=np.ones((50,1),dtype=int),values=X,axis=1) 

# =============================================================================
# PASO 1: SELECCIONAR EL NIVEL DE SIGNIFICACION
SL=0.05
# =============================================================================


def backwardElimination(x, sl):
# =============================================================================
# Eliminación hacia atrás utilizando solamente p-valores
# =============================================================================
    numVars = len(x[0])
    for i in range(0, numVars):
# =============================================================================
# PASO 2: SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORAS
        regressor_OLS = sm.OLS(y,x).fit()
# =============================================================================
# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
        maxVar = max(regressor_OLS.pvalues).astype(float)
        # SI P>SL, ENTONCES VAMOS AL PASO 4.
        if maxVar > sl:
            # =================================================================
            # PASO 4: SE ELIMINA LA VARIABLE PREDICTORA
            for j in range(0, numVars - i):
                # Buscando de nuevo el p-valor a eliminar para dar con la 
                # columna a eliminar.
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    # eliminando la columna j de la tabla x. 
                    x = np.delete(x, j, 1)
            # =================================================================
        # =====================================================================
        # PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE.(PASO 2)
        # =====================================================================
    # SINO AL FIN.
    print(regressor_OLS.summary())
    return x 
# =============================================================================

# matriz de caracteristicas optimas
# variables significativas (empezamos con todas las variables)
X_opt=X[:,[0,1,2,3,4,5]]
X_Modeled = backwardElimination(copy.deepcopy(X_opt), SL)

def backwardElimination2(x, sl):
# =============================================================================
#Eliminación hacia atrás utilizando p-valores y el valor de R Cuadrado Ajustado
# =============================================================================
    numVars = len(x[0])
    temp= np.zeros((50,6),dtype=int)
    for i in range(0, numVars):
# =============================================================================
# PASO 2: SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORAS
        regressor_OLS = sm.OLS(y,x).fit()
# =============================================================================
# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before=regressor_OLS.rsquared_adj.astype(float)
        # SI P>SL, ENTONCES VAMOS AL PASO 4.
        if maxVar > sl:
            # =================================================================
            # PASO 4: SE ELIMINA LA VARIABLE PREDICTORA
            for j in range(0, numVars - i):
                # Buscando de nuevo el p-valor a eliminar para dar con la 
                # columna a eliminar.
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j]=x[:,j]
                    # eliminando la columna j de la tabla x. 
                    x = np.delete(x, j, 1)
                    
                    tmp_regressor = sm.OLS(y,x).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue 
            # =================================================================
        # =====================================================================
        # PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE.(PASO 2)
        # =====================================================================
    # SINO AL FIN.
    print(regressor_OLS.summary())
    return x 
# =============================================================================


X_Modeled2 = backwardElimination2(copy.deepcopy(X_opt), SL)
