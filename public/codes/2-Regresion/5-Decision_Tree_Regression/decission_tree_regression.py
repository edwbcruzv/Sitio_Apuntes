# -*- coding: utf-8 -*-

# =============================================================================
# Decission Tree Regression
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
# Ajustar la regresion con arboles de decision con el dataset
# =============================================================================

from sklearn.tree import DecisionTreeRegressor
tree_regression=DecisionTreeRegressor(random_state=0)
tree_regression.fit(X,y)

y_pred=tree_regression.predict(X)
# =============================================================================
# Visualizacion de los resultado: Arboles de Decision
# =============================================================================

# Del minimo de la variable independiente en sumandole 0.1 hasta el maximo
X_grid =np.arange(min(X),max(X),0.01)
#
# se ajusta a una matriz
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,tree_regression.predict(X_grid),color='green') # linea 1
#plt.plot(X,tree_regression.predict(X),color='green') # linea 2
plt.title("Modelo Regresion por arboles de decision")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()

# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred=tree_regression.predict([[6.5]])
print(y_pred)
