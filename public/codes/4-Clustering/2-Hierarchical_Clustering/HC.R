# =============================================================================
# Clustering Jerarquico
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
# ----Utilizando Dendrogramas para encontrar el numero optimo de cluster--
# =============================================================================
dendrogram=hclust(dist(X,method = "euclidean"),
                  method = "ward.D")
plot(dendrogram,
     main = "Dendrograma",
     xlab = "clientes del centro comercial",
     ylab = "distancia euclidea")
# =============================================================================
# ----------Aplicando el metodo Kmeans para segmentar el dataset---------
# =============================================================================
# Con ayuda de la grafica anterior sabemos el el k = 5 es el optimo
# k_means=kmeans(X,5,iter.max = 300, nstart = 10)

# =============================================================================
# ----Ajustando el clustering jerarquico a los datos----
# =============================================================================

hc=hclust(dist(X,method = "euclidean"),
                  method = "ward.D")
y_hc = cutree(hc,k=5,)
# =============================================================================
# Visualizacion de los datos
# =============================================================================
# library(cluster)

clusplot(X, y_hc, lines = 0, shade = TRUE, 
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion")








