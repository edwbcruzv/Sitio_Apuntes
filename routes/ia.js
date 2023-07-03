let express = require('express');
let router = express.Router();


router.get('/definicion', function (req, res, next) {
    res.render('ia/definicion');
});

router.get('/agentes', function (req, res, next) {
    res.render('ia/agentes');
});

router.get('/busqueda', function (req, res, next) {
    res.render('ia/busqueda');
});

router.get('/logica', function (req, res, next) {
    res.render('ia/logica');
});


module.exports = router;