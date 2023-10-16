# =============================================================================
# Regresion Lineal Multiple
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

# Estructura de los datos:
# Contiene informacion de 50 empresas y sus gastos en 3 areas, lugar 
# donde se ubican y la ganancia 
# Filas :50
# Columnas:
#           |RyD Spend|Administration|Marketing Spend|State (vars independiente)
#           |Profit (var_dependiente)
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
# Eliminación hacia atrás utilizando solamente p-valores
# =============================================================================
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

dataset = dataset[, c(1,2,3,4,5)]
print(backwardElimination(training_set, SL))
