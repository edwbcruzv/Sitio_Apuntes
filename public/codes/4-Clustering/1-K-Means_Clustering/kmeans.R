# =============================================================================
# K-Means
# =============================================================================
# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

dataset = read.csv('Mall_Customers.csv')
X=dataset[,4:5]

# No se hace ninguna distincion entre variables independientes 
# y variables dependientes en R
# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================
# No aplica
# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# No aplica
# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings------------
# =============================================================================
# No aplica
# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================
# No aplica
# =============================================================================
# ---------Metodo del codo para averiguarl el numero optimo de clusters-------
# =============================================================================
set.seed(0)
wcss=vector()
for(i in 1:10){
  wcss[i] <- sum(kmeans(X,i)$withinss )
}
print(wcss)
plot( 1:10, wcss, type = 'b', main= "Metodo del codo",
     xlab = "Numero de clusters K",
     ylab = "WCSS K")

# =============================================================================
# ----------Aplicando el metodo Kmeans para segmentar el dataset---------
# =============================================================================
# Con ayuda de la grafica anterior sabemos el el k = 5 es el optimo
k_means=kmeans(X,5,iter.max = 300, nstart = 10)

# =============================================================================
# ---------------Visualizacionde los clusters------------
# =============================================================================
# library(cluster)
clusplot(X, k_means$cluster, lines = 0, shade = TRUE, 
        color = TRUE,
        labels = 4,
        plotchar = FALSE,
        span = TRUE,
        main = "Clustering de clientes",
        xlab = "Ingresos anuales",
        ylab = "Puntuacion")








