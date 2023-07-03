let express = require('express');
let router = express.Router();



router.get('/requisitos', function (req, res, next) {
    res.render('dl/requisitos');
});

router.get('/rna', function (req, res, next) {
    res.render('dl/rna');
});

router.get('/rnc', function (req, res, next) {
    res.render('dl/rnc');
});

router.get('/rnr', function (req, res, next) {
    res.render('dl/rnr');
});

router.get('/mapas_auto', function (req, res, next) {
    res.render('dl/mapas_auto');
});

module.exports = router;