#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:57:19 2023

@author: cruz
"""
# =============================================================================
# Clustering Jerarquico
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
# ----Utilizando Dendrogramas para encontrar el numero optimo de cluster--
# =============================================================================
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))

plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidia")
plt.show()

# =============================================================================
# ----Ajustando el clustering jerarquico a los datos----
# =============================================================================
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_pred=hc.fit_predict(X)



plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=50, c="red",label = "Cluster1")
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=50, c="blue",label = "Cluster2")
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=50, c="green",label = "Cluster3")
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=50, c="cyan",label = "Cluster4")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=50, c="magenta",label = "Cluster5")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales en miles de $")
plt.ylabel("Puntuacion de gastos")
plt.legend()
plt.show()






