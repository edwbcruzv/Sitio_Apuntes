#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:37:22 2023

@author: cruz
"""
# =============================================================================
# K Means
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
# los clientes de un centro comercial, son membrecias de clientes
# historial del gasto y puntos de una membresia, edad y genero
# objetivo: segmentar a los clientes y obetener una conclusion
dataset = pd.read_csv('Mall_Customers.csv')

# Variables independiente:Mayuscula por ser una matriz.
X = dataset.iloc[:,[3,4]].values

# Variable dependiente:minuscula por ser un vector.
# y = dataset.iloc[:,-1].values 

# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================
# No hace falta
# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# No hace falta
# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings------------------
# =============================================================================
# No hace falta
# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================
# No hace falta

# =============================================================================
# ---------Metodo del codo para averiguarl el numero optimo de clusters-------
# =============================================================================
from sklearn.cluster import KMeans

wcss = []
# ejecutaremos 10 veces con 1 hasta 10 clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init= "k-means++", max_iter = 300, n_init= 10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # Suma de los cuadrados (distancias)

    
plt.plot(range(1,11), wcss)
plt.title("Metodo del Codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS")
plt.show()

# =============================================================================
# ----------Aplicando el metodo Kmeans para segmentar el dataset---------
# =============================================================================
# Con ayuda de la grafica anterior sabemos el el k = 5 es el optimo

kmeans = KMeans(n_clusters=5,
                init="k-means++", # inicializacion de los varicentros (no aleatoria)
                max_iter=300, # numero maximo de iteraciones
                n_init=10, # inicializacion aleatoria
                random_state=0)
y_pred = kmeans.fit_predict(X)


# =============================================================================
# ---------------Visualizacionde los clusters------------
# =============================================================================

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=50, c="red",label = "Cluster1")
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=50, c="blue",label = "Cluster2")
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=50, c="green",label = "Cluster3")
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=50, c="cyan",label = "Cluster4")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=50, c="magenta",label = "Cluster5")

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c="yellow", label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales en miles de $")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()


