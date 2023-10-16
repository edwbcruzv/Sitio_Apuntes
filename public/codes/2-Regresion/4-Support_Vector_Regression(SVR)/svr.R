# =============================================================================
#  Support Vectorial Regression
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]
# =============================================================================
# Ajustar la SVR con el dataset
# =============================================================================
# install.packages("e1071") # solo se necesita ejecutar una vez
# library(e1071)
svr_regressor=svm(x = dataset$Level,y = dataset$Salary,kernel='radial')
summary(svr_regressor)

# =============================================================================
# Visualizacion de los resultado: Modelo {type}
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento

# grid=seq(min(dataset$Level),max(dataset$Level),0.1)
# Agregando componentes a mostrar
ggplot()+
  # Dobujando los puntos de entrenamiento
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  # Dobujando la linea de la prediccion, en base al entrenamiento
  geom_line(aes(x=dataset$Level,y=predict(svr_regressor,newdata=dataset$Level)),
            color="blue")+
  ggtitle("Prediccion SVR  ")+
  xlab("Nivel del empleado")+
  ylab("Sueldo en $")

# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred_type=predict(svr_regressor,newdata=6.5)
print(y_pred_type)

