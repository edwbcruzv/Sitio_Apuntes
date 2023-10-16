# =============================================================================
# --------------------Plantilla de Pre-procesado--------------------
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================
# Estructura de los datos: {explicar eldataset y el objetivo}.
# Filas :{numero de filas}
# Columnas:
#           |{col1}|{col2}|{...} (vars independiente)
#           |{columna de var indep.}| (var_dependiente)
dataset = read.csv('dataset.csv')

# No se hace ninguna distincion entre variables independientes 
# y variables dependientes en R

# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings-------------
# =============================================================================
# {se pueden modificar segun se necesite}
# =============================================================================
# Ajustar la regresion {sea cualquier tipo} con el dataset
# =============================================================================
# crear nuestra variable de regresion aqui
type_regressor=lm(formula=Salary ~ .,data=dataset)
summary(type_regressor)
# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred_type=predict(type_regressor,newdata=testing_data)
print(y_pred_type)
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
  geom_line(aes(x=dataset$Level,y=predict(type_regressor,newdata=dataset)),
            color="blue")+
  ggtitle("Prediccion {type}  ")+
  xlab("labelx")+
  ylab("labely")


