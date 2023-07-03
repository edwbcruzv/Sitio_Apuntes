let express = require('express');
let router = express.Router();


router.get('/requisitos', function (req, res, next) {
    res.render('ml/requisitos');
});

router.get('/preprocesado', function (req, res, next) {
    res.render('ml/preprocesado');
});

router.get('/regresion', function (req, res, next) {
    res.render('ml/regresion');
});

router.get('/clasificacion', function (req, res, next) {
    res.render('ml/clasificacion');
});

router.get('/clustering', function (req, res, next) {
    res.render('ml/clustering');
});

router.get('/a_r_aprendizaje', function (req, res, next) {
    res.render('ml/a_r_aprendizaje');
});

router.get('/a_reforzado', function (req, res, next) {
    res.render('ml/a_reforzado');
});

router.get('/pln', function (req, res, next) {
    res.render('ml/pln');
});

router.get('/r_dimensiones', function (req, res, next) {
    res.render('ml/r_dimensiones');
});

router.get('/boosting', function (req, res, next) {
    res.render('ml/boosting');
});

module.exports = router;