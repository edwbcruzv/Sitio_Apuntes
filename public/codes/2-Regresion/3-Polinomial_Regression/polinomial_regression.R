# =============================================================================
# --------------------Plantilla de Pro-procesado--------------------
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

dataset = read.csv('Position_Salaries.csv')
dataset=dataset[,2:3]

# No se hace ninguna distincion entre variables independientes 
# y variables dependientes en R
# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings-------------
# =============================================================================

# No es recomendable dividir los datos, al ser pocos a simple vista necesita
# toda la informacion.

# =============================================================================
# Ajustar la regresion lineal con el dataset
# =============================================================================
linear_regressor=lm(formula=Salary ~ .,data=dataset)
summary(linear_regressor)
# =============================================================================
# Ajustar la regresion polinomica con el dataset
# =============================================================================
dataset$Level2=dataset$Level^2 # agregando esto se modifica el valor a polinomica
dataset$Level3=dataset$Level^3
dataset$Level4=dataset$Level^4
dataset$Level5=dataset$Level^5
poly_regressor=lm(formula=Salary ~ .,data=dataset)
summary(poly_regressor)
# =============================================================================
# Visualizacion de los resultado: Modelo Lineal
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento y la recta del modelo lineal

# Agregando componentes a mostrar
ggplot()+
  # Dobujando los puntos de entrenamiento
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  # Dobujando la linea de la prediccion, en base al entrenamiento
  geom_line(aes(x=dataset$Level,y=predict(linear_regressor,newdata=dataset)),
            color="blue")+
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado ")+
  xlab("Nivel empleado")+
  ylab("Sueldo en $")
# =============================================================================
# Visualizacion de los resultado: Modelo Polinomico
# =============================================================================
# Para mostrar la grafica de datos de entrenamiento y la recta del modelo polinomial

# Agregando componentes a mostrar
ggplot()+
  # Dobujando los puntos de entrenamiento
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             color="red")+
  # Dobujando la linea de la prediccion, en base al entrenamiento
  geom_line(aes(x=dataset$Level,y=predict(poly_regressor,newdata=dataset)),
            color="blue")+
  ggtitle("Prediccion lineal polinomial del sueldo en funcion del nivel del empleado ")+
  xlab("Nivel empleado")+
  ylab("Sueldo en $")
# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred_linear=predict(linear_regressor,newdata=data.frame(Level=6.5))
print(y_pred_linear)

y_pred_poly=predict(poly_regressor,newdata=data.frame(Level=6.5,
                                                      Level2=6.5^2,
                                                      Level3=6.5^3,
                                                      Level4=6.5^4))
print(y_pred_poly)
