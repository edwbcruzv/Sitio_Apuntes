#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Plantilla de Clasificacion Naive Bayes
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

dataset = pd.read_csv('Social_Network_Ads.csv')

# Variable independiente:Mayuscula por ser una matriz.
X = dataset.iloc[:,2:4].values

# Variable dependiente:minuscula por ser un vector.
y = dataset.iloc[:,-1].values

# Nota: convertir a matrices tanto a X como a y para evitar errores
# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================

from sklearn.impute import SimpleImputer
# Los valores desconocidos de los valores independientes son los NA´s.
# El valor que se va a sustituir que sera la media.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Seleccionando las columnas las cuales estan los valores NA´s.
# [Todas las filas,Columnas 1 y 2]
imputer = imputer.fit(X[:,1:3]) # ajustando valores
# Sobreescribirnedo la matriz con la nueva trasformacion configurada.
X[:,1:3] = imputer.transform(X[:,1:3])

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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Escalador para las variables independientes
sc_X = StandardScaler()
# escalando variables de training, se usa el fit_trasform
X_train = sc_X.fit_transform(X_train)
# Se escala con el mismo escalador con las variables de testing con transform
# para que la trasformacion lo haga en base al conjunto escalado de training
X_test = sc_X.transform(X_test)

## En este caso no es necesario escalar las variables dependiente,
## pero en otras ocaciones si se necesitaran escalar

# =============================================================================
# Ajustar el modelo {modelo de clasificacion} al conjunto de entrenamiento
# =============================================================================
from sklearn.naive_bayes import GaussianNB 

classifier = GaussianNB()
classifier.fit(X_train,y_train)

# =============================================================================
# Prediccion de los resultados con el conjunto de testing
# =============================================================================

y_pred=classifier.predict(X_test)



# =============================================================================
# Elaborar una Matriz de confusion
# 
# |----------------------|----------------------|
# |     Verdaderos       |      Falsos          |
# |     Positivo         |      Positivos       |
# |----------------------|----------------------|
# |     Falsos           |      Verdaderos      |
# |     Negativos        |      Negativos       |
# |----------------------|----------------------|
# =============================================================================

from sklearn.metrics import confusion_matrix

c_m=confusion_matrix(y_test, y_pred)

# =============================================================================
# Representacion grafica de los resultados del modelo (Entrenamiento)
# =============================================================================

from matplotlib.colors import ListedColormap
X_set, y_set=X_train, y_train

# Genera todos los punto del dominio posible (mallado de la region)
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
# pindando todo el plano 
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
# division de la region del trabajo (Dominio)
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

# pintando los resultados categoricos en el grafico
for i, j in enumerate(np.unique(y_set)):
    aux=(y_set == j).ravel()
    plt.scatter(X_set[aux, 0], X_set[aux, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Modelo Naive Bayes Entrenamiento')
plt.xlabel('algo en x')
plt.ylabel('Si/No')
plt.legend()
plt.show()


# =============================================================================
# Representacion grafica de los resultados del modelo (Testing)
# =============================================================================

X_set, y_set=X_test, y_test

# Genera todos los punto del dominio posible (mallado de la region)
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
# pindando todo el plano 
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
# division de la region del trabajo (Dominio)
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

# pintando los resultados categoricos en el grafico
for i, j in enumerate(np.unique(y_set)):
    aux=(y_set == j).ravel()
    plt.scatter(X_set[aux, 0], X_set[aux, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Modelo Naive Bayes Testing')
plt.xlabel('algo en x')
plt.ylabel('Si/No')
plt.legend()
plt.show()