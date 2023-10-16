
# =============================================================================
# Plantilla de Clasificacion
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

# Estructura de los datos: {explicar eldataset y el objetivo}.
# Filas :{numero de filas}
# Columnas:
#           |{col1}|{col2}|{...} (vars independiente)
#           |{columna de var indep.}| (var_dependiente)
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

# No se hace ninguna distincion entre variables independientes 
# y variables dependientes en R

# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings-----------------
# =============================================================================
# install.packages("caTools") # solo se necesita ejecutar una vez
# library(caTools)
# configurando semilla aleatoria para la division de datos
set.seed(0)
# se elige el porcentaje de los datos para el training en %
split = sample.split(dataset$Purchased,SplitRatio = 0.25)
print(split)
# Dividiendo el conjunto , False para el test
training_set = subset(dataset,split == FALSE)
# Dividiendo el conjunto , True para el training
testing_set = subset(dataset,split == TRUE)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================
# Scale() necesitaremos especificar las filas y columnas
# para especificar cuales son variables numericas.
# Ya que factor() no sobreescribe el dataset
training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])

# =============================================================================
# Ajustar el modelo de {  } al conjunto de entrenamiento
# =============================================================================

classifier =
  
  # =============================================================================
# Prediccion de los resultados con el conjunto de testing
# =============================================================================
# obtenemos las probabilidades listadas
prob_pred = predict(classifier,type = "response", newdata = testing_set[,-3])
print(prob_pred)
# 

y_pred = ifelse(prob_pred > 0.5, 1, 0)
print(y_pred)

# =============================================================================
# Elaborar una Matriz de confusion
# 
# |----------------------|----------------------|
# |     Verdaderos       |      Falsos          |
# |     Positivo         |      Positivos       |
# |----------------------|----------------------|
# |     Falsos           |      Verdaderos      |
# |     Negativos        |      Negativos       |
# |----------------------|----------------------|
# =============================================================================

cm = table(testing_set[,3],y_pred)
print(cm)

# =============================================================================
# Representacion grafica de los resultados del modelo (Entrenamiento)
# =============================================================================

# install.packages("ElemStatLearn") # solo se necesita ejecutar una vez
# library(ElemStatLearn)

set = training_set
X1 = seq(min(set[,1]) -1, max(set[,1]) + 1, by = 0.01)
X2 = seq(min(set[,2]) -1, max(set[,2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age','EstimatedSalary')
prob_set=predict(classifier,type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

plot(
  set[,-3],
  main = "Clasificacion Bosques Aleatorios (Conjunto de Entrenamiento)",
  xlab = 'Edad',
  ylab = 'Sueldo Estimado',
  xlim = range(X1),
  ylim = range(X2))

contour(
  X1,
  X2,
  matrix(as.numeric(y_grid), length(X1), length(X2)),
  add = TRUE)

points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))

# =============================================================================
# Representacion grafica de los resultados del modelo (Testing)
# =============================================================================

set = testing_set
X1 = seq(min(set[,1]) -1, max(set[,1]) + 1, by = 0.01)
X2 = seq(min(set[,2]) -1, max(set[,2]) + 1, by = 0.01)
grid_set = expand.grid(X1,X2)
colnames(grid_set) = c('Age','EstimatedSalary')
prob_set=predict(classifier, type = "response", newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

plot(
  set[,-3],
  main = "Clasificacion Bosques Aleatorios (Conjunto de Testing)",
  xlab = 'Edad',
  ylab = 'Sueldo Estimado',
  xlim = range(X1),
  ylim = range(X2))

contour(
  X1,
  X2,
  matrix(as.numeric(y_grid), length(X1), length(X2)),
  add = TRUE)

points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))





