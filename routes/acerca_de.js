let express = require('express');
let router = express.Router();


router.get('/', function (req, res, next) {
    res.render('acerca_de');
});

module.exports = router;