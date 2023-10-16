# =============================================================================
# Random Forest Regression
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
# Ajustar la regresion random forest con el dataset
# =============================================================================
# crear nuestra variable de regresion aqui
# library(randomForest)
set.seed(1)
rf_regressor=randomForest(x=dataset[1],y=dataset$Salary,
                          ntree=1000)
summary(rf_regressor)

# =============================================================================
# Visualizacion de los resultado: Modelo random forest
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento

x_grid=seq(min(dataset$Level),max(dataset$Level),0.01)
# Agregando componentes a mostrar
ggplot()+
  # Dobujando los puntos de entrenamiento
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  # Dobujando la linea de la prediccion, en base al entrenamiento
  # geom_line(aes(x=dataset$Level,y=predict(rf_regressor,newdata=dataset)),
            geom_line(aes(x=x_grid,
                          y=predict(rf_regressor,
                                    newdata=data.frame(Level=x_grid))),
            color="blue")+
  ggtitle("Prediccion usando regresion con arboles aleatorios de decision   ")+
  xlab("labelx")+
  ylab("labely")

# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred_tree=predict(rf_regressor,newdata=data.frame(Level=6.5))
print(y_pred_tree)

