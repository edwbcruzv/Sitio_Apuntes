let express = require("express");
let fs = require("fs")
let router = express.Router();


function GetCode(path)  {
    console.log(path)
    try {
        // Intenta leer el contenido del archivo .py
        return fs.readFileSync(path, "utf-8");
      } catch (error) {
        // Captura y maneja cualquier error que ocurra al leer el archivo
        return "Error: No se encontró el archivo .py en la ruta especificada.";
      }
}


router.get("/preprocesado", function (req, res, next) {
    let prep_py = GetCode("public/codes/1-Data_Preprocesing/data_preprocessing_template.py")
    let prep_r= GetCode("public/codes/1-Data_Preprocesing/data_preprocessing_template.R")
  res.render("ml/preprocesado",{
    Preprocesado_Python:prep_py,
    Preprocesado_R:prep_r
  });
});

router.get("/regresion", function (req, res, next) {
    let regresion_lineal_python = GetCode("public/codes/2-Regresion/1-Simple_Linear_Regression/simple_linear_regression.py")
    let regresion_lineal_r= GetCode("public/codes/2-Regresion/1-Simple_Linear_Regression/simple_linear_regression.R")
    let regresion_Multiple_python= GetCode("public/codes/2-Regresion/2-Multiple_Linear_Regression/multiple_linear_regression.py")
    let regresion_Multiple_r= GetCode("public/codes/2-Regresion/2-Multiple_Linear_Regression/multiple_linear_regression.R")
    let regresion_polinomica_python= GetCode("public/codes/2-Regresion/3-Polinomial_Regression/polinomial_regression.py")
    let regresion_polinomica_r= GetCode("public/codes/2-Regresion/3-Polinomial_Regression/polinomial_regression.R")
    let regresion_svm_python= GetCode("public/codes/2-Regresion/4-Support_Vector_Regression(SVR)/svr.py")
    let regresion_svm_r= GetCode("public/codes/2-Regresion/4-Support_Vector_Regression(SVR)/svr.R")
    let regresion_arboles_python= GetCode("public/codes/2-Regresion/5-Decision_Tree_Regression/decission_tree_regression.py")
    let regresion_arboles_r= GetCode("public/codes/2-Regresion/5-Decision_Tree_Regression/decission_tree_regression.R")
    let regresion_bosques_python= GetCode("public/codes/2-Regresion/6-Random_Forest_Regression/random_forest_regression.py")
    let regresion_bosques_r= GetCode("public/codes/2-Regresion/6-Random_Forest_Regression/random_forest_regression.R")

  

  res.render("ml/regresion", {
    Regresion_Lineal_Python: regresion_lineal_python,
    Regresion_Lineal_R: regresion_lineal_r,
    Regresion_Multiple_Python: regresion_Multiple_python,
    Regresion_Multiple_R: regresion_Multiple_r,
    Regresion_Polinomica_Python: regresion_polinomica_python,
    Regresion_Polinomica_R: regresion_polinomica_r,
    Regresion_SVM_Python: regresion_svm_python,
    Regresion_SVM_R: regresion_svm_r,
    Regresion_Arboles_Python: regresion_arboles_python,
    Regresion_Arboles_R: regresion_arboles_r,
    Regresion_Bosques_Python: regresion_bosques_python,
    Regresion_Bosques_R: regresion_bosques_r,
  });
});

router.get("/clasificacion", function (req, res, next) {
  res.render("ml/clasificacion",{
    Clasificacion_Plantilla_Py:GetCode("public/codes/3-Classification/1-Logistic_Regression/classification_template.py"),
    Clasificacion_Plantilla_R:GetCode("public/codes/3-Classification/1-Logistic_Regression/classification_template.R"),
    Clasificacion_RLog_Py:GetCode("public/codes/3-Classification/1-Logistic_Regression/logistic_regression.py"),
    Clasificacion_RLog_R:GetCode("public/codes/3-Classification/1-Logistic_Regression/logistic_regression.R"),
    Clasificacion_KNN_Py:GetCode("public/codes/3-Classification/2-K-Nearest_Neighbors(K-NN)/knn.py"),
    Clasificacion_KNN_R:GetCode("public/codes/3-Classification/2-K-Nearest_Neighbors(K-NN)/knn.R"),
    Clasificacion_SVM_Py:GetCode("public/codes/3-Classification/3-Support_Vector_Machine(SVM)/svm.py"),
    Clasificacion_SVM_R:GetCode("public/codes/3-Classification/3-Support_Vector_Machine(SVM)/svm.R"),
    Clasificacion_KernelSVM_Py:GetCode("public/codes/3-Classification/4-Kernel_SVM/kernel-svm.py"),
    Clasificacion_KernelSVM_R:GetCode("public/codes/3-Classification/4-Kernel_SVM/kernel-svm.R"),
    Clasificacion_Naive_Py:GetCode("public/codes/3-Classification/5-Naive_Bayes/naive_bayes.py"),
    Clasificacion_Naive_R:GetCode("public/codes/3-Classification/5-Naive_Bayes/naive_bayes.R"),
    Clasificacion_Arboles_Py:GetCode("public/codes/3-Classification/6-Decision_Tree_Classification/decision_tree_classification.py"),
    Clasificacion_Arboles_R:GetCode("public/codes/3-Classification/6-Decision_Tree_Classification/decision_tree_classification.R"),
    Clasificacion_Bosques_Py:GetCode("public/codes/3-Classification/7-Random_Forest_Classification/random_forest_classification.py"),
    Clasificacion_Bosques_R:GetCode("public/codes/3-Classification/7-Random_Forest_Classification/random_forest_classification.R"),

  });
});

router.get("/clustering", function (req, res, next) {
  res.render("ml/clustering",{
    Cluster_KMeans_Py:GetCode("public/codes/4-Clustering/1-K-Means_Clustering/kmeans.py"),
    Cluster_KMeans_R:GetCode("public/codes/4-Clustering/1-K-Means_Clustering/kmeans.R"),
    Cluster_HC_Py:GetCode("public/codes/4-Clustering/2-Hierarchical_Clustering/HC.py"),
    Cluster_HC_R:GetCode("public/codes/4-Clustering/2-Hierarchical_Clustering/HC.R")
  });
});

router.get("/a_r_aprendizaje", function (req, res, next) {
  res.render("ml/a_r_aprendizaje");
});

router.get("/a_reforzado", function (req, res, next) {
  res.render("ml/a_reforzado");
});

router.get("/pln", function (req, res, next) {
  res.render("ml/pln");
});

router.get("/r_dimensiones", function (req, res, next) {
  res.render("ml/r_dimensiones");
});

router.get("/boosting", function (req, res, next) {
  res.render("ml/boosting");
});

module.exports = router;
