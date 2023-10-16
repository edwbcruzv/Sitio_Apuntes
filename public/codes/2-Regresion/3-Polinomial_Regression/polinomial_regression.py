# =============================================================================
# Regresion Lineal Polinomica
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

dataset = pd.read_csv('Position_Salaries.csv')

# Variable independiente:Mayuscula por ser una matriz.
X = dataset.iloc[:,1:2].values

# Variable dependiente:minuscula por ser un vector.
y = dataset.iloc[:,2:3].values

# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings-------------
# =============================================================================

# No es recomendable dividir los datos, al ser pocos a simple vista necesita
# toda la informacion.

# =============================================================================
# Ajustar la regresion lineal con el dataset
# =============================================================================
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X,y)

# =============================================================================
# Ajustar la regresion polinomica con el dataset
# =============================================================================

from sklearn.preprocessing import PolynomialFeatures

poly_regression=PolynomialFeatures(degree=3) # se puede jugar con el grado
X_poly=poly_regression.fit_transform(X)

polynomial_regression=LinearRegression()
polynomial_regression.fit(X_poly,y)

# =============================================================================
# Visualizacion de los resultado: Modelo Lineal
# =============================================================================
plt.scatter(X,y,color='red')
plt.plot(X,linear_regression.predict(X),color='blue')
plt.title("Modelo Regresion Lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()
# =============================================================================
# Visualizacion de los resultado: Modelo Polinomico
# =============================================================================
plt.scatter(X,y,color='red')
plt.plot(X,polynomial_regression.predict(X_poly),color='green')
plt.title("Modelo Regresion Polinomica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()
# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================

linear_regression.predict([[6.5]])
polynomial_regression.predict(poly_regression.fit_transform([[6.5]]))















