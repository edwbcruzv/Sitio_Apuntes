# =============================================================================
# Decission Tree Regression
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================
# Estructura de los datos:tipos de empleados y el nivel 
# de cada tipo de empleado y el salario correspondiente.
# Objetivo: asignar un salario correspondientes a un nivel y su posicion dada.
# Filas :10
# Columnas:
#           |Position|Level| (vars independiente)
#           |salary| (var_dependiente)
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

# No se hace ninguna distincion entre variables independientes 
# y variables dependientes en R

# =============================================================================
# Ajustar la regresion con el dataset
# =============================================================================
# crear nuestra variable de regresion aqui
# library(rpart)
tree_regressor=rpart(formula = dataset$Salary ~ dataset$Level,
                     data=dataset,
                     control = rpart.control(minsplit = 1))
summary(tree_regressor)

# =============================================================================
# Visualizacion de los resultado: Modelo {type}
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento

x_grid=seq(min(dataset$Level),max(dataset$Level),0.01)
# Agregando componentes a mostrar
ggplot()+
  # Dobujando los puntos de entrenamiento
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  # Dobujando la linea de la prediccion, en base al entrenamiento
  geom_line(aes(x=dataset$Level,y=predict(tree_regressor,newdata=dataset)),
  # geom_line(aes(x=x_grid,
  #               y=predict(tree_regressor,
  #                         newdata=data.frame(Level=x_grid))), # ver porque no funciona esta parte----
            color="blue")+
  ggtitle("Prediccion usando regresion con arboles de decision   ")+
  xlab("labelx")+
  ylab("labely")

# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred_tree=predict(tree_regressor,newdata=data.frame(Level=6.5))
print(y_pred_tree)

