# =============================================================================
# Regresion Lineal Multiple
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

dataset = read.csv('50_Startups.csv')

# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================

# se convierte una columna en factores
dataset$State = factor(dataset$State,
                         # dandole un valos a cada etiqueta dentro de la columna
                         levels = c("New York","California", "Florida"),
                         # etiquetas
                         labels = c(1,2,3))


# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings------------
# =============================================================================
# install.packages("caTools") # solo se necesita ejecutar una vez
# library(caTools)

# configurando semilla aleatoria para la division de datos
set.seed(123)
# se elige el porcentaje de los datos para el training en %
split = sample.split(dataset$Profit,SplitRatio = 0.8)

# Dividiendo el conjunto , False para el test
training_set = subset(dataset,split == TRUE)
# Dividiendo el conjunto , True para el training
testing_set = subset(dataset,split == FALSE)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================

# no se necesita escalado

# =============================================================================
# --------------------Ajustando el modelo de Regresion Lineal Multiple---------
# --------------------con el conjunto de entrenamiento--------------------
# =============================================================================

# con el punto indicamos que el reto de columnas se usaran como variables 
# independientes
regression=lm(formula=Profit ~ .,data=training_set)
summary(regression)

# la libreria lm resolvera el problema de la multicolinealidad (var Dummy)
# checar los '*' en signif. codes, para visualizar 
# la significatividad estadistica, entre mas '*' mejor. 

# =============================================================================
# --------------------Predecir el conjunto de testing--------------------
# =============================================================================
# (modelo de prediccion,datos de testing)
y_pred=predict(regression,newdata=testing_set)
print(y_pred)

# =============================================================================
# -----------Ajustando el modelo optimo de Regresion Lineal Multiple-----------
# 
# Eliminacion hacia atras
# Usando todas las variables independientes (las dummy son automaticas)
# =============================================================================

# =============================================================================
# PASO 1: SELECCIONAR EL NIVEL DE SIGNIFICACION
SL=0.05
# =============================================================================

# =============================================================================
# PASO 2: SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORAS
# matriz de caracteristicas optimas
# variables significativas (empezamos con todas las variables)
regression=lm(formula=Profit ~ R.D.Spend +
                              Administration + 
                              Marketing.Spend +
                              State,
                              data=dataset)
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
summary(regression)
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que State2 > SL y State3 > SL, 
      # entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA state, 
# para eliminar ambas que son altas.
# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression=lm(formula=Profit ~ R.D.Spend +
                Administration + 
                Marketing.Spend,
              data=dataset)
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
summary(regression)
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que Administration > SL, 
# entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA state, 
# para eliminar ambas que son altas.
# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression=lm(formula=Profit ~ R.D.Spend + 
                Marketing.Spend,
              data=dataset)
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
summary(regression)
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que Marketing.Spend > SL, 
# entonces vamos al PASO 4.
# SINO AL FIN.
# =============================================================================

# =============================================================================
# PASO 4: SE ELIMINA LA VARIABLE PREDICTORA state, 
# para eliminar ambas que son altas.
# =============================================================================
# PASO 5: AJUSTAR EL MODELO SIN DICHA VARIABLE
regression=lm(formula=Profit ~ R.D.Spend ,
              data=dataset)
# =============================================================================

# =============================================================================
# PASO 3: CONSIDERAR LA VARIABLE PREDICTORA CON EL P-VALOR MAS GRANDE.
summary(regression)
# SI P>SL, ENTONCES VAMOS AL PASO 4.
# Al mostrar con sumary vemos que no hay variables > SL, 
# entonces vamos al PASO 4.
# SINO AL FIN. y llegamos a un modelo lineal simple
# =============================================================================