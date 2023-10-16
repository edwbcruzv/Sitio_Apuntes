# -*- coding: utf-8 -*-

# --------------------Regresion lineal simple--------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

# Dataset de empleados que incluye los años de esperiencia de un empleado
# y de cuanto es su salario.
# Columnas: |YearExperience| (var independiente)
#           |Salary (var. dependiente)
dataset = pd.read_csv('Salary_Data.csv')
# Se toman las primeras columnas, menos la ultima.
X = dataset.iloc[:,:-1].values 
# Se toma la ultima columna.
y = dataset.iloc[:,1].values

# =============================================================================
# --------------------Dividir el dataset en conjunto--------------------
# --------------------de entrenamiento y conjunto de testing--------------------
# =============================================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=2)

# =============================================================================
# --------------------Crear modelo de Regresion Lineal--------------------
# --------------------Simple con el conjunto de entrenamiento--------------------
# =============================================================================
from sklearn.linear_model import LinearRegression

# Modulo que crea los modelos de regresion lineal
regression = LinearRegression()
# (Conjunto de variables independientes, Conjunto de variables a predecir)
regression.fit(X_train ,y_train)
# =============================================================================
# --------------------Predecir el conjunto de test--------------------
# =============================================================================
# y_pred nos entrega el modelo lineal, en base al conjunto de testing.
# Al visualizar ambas tablas los valores deberian de ser casi similares.
y_pred = regression.predict(X_test)

# =============================================================================
# --------------------Visualizar los resultados de entrenamiento--------------------
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento y la recta del modelo lineal

# Dubujando los puntos del conjunto de entrenamiento de color rojo.
plt.scatter(X_train,y_train, color="red")
# Dobujando la linea de regresion de color azul
plt.plot(X_train,regression.predict(X_train),color="blue")
# Dando etiquetas a la grafica
plt.title("Sueldo vs Años de experiencia(Entrenamiento)")
plt.ylabel("Sueldo ($)")
plt.xlabel("Anos de experiencia")
plt.show()

# =============================================================================
# --------------------Visualizar los resultados de test--------------------
# =============================================================================
# Para mostrar la grafica de datos de testing y la recta del modelo lineal

# Dubujando los puntos del conjunto de testing de color rojo.
plt.scatter(X_test,y_test, color="red")
# Dobujando la linea de regresion de color azul
plt.plot(X_train,regression.predict(X_train),color="blue")
# Dando etiquetas a la grafica
plt.title("Sueldo vs Años de experiencia(Testing)")
plt.ylabel("Sueldo ($)")
plt.xlabel("Anos de experiencia")
plt.show()

