let express = require('express');
let router = express.Router();


router.get('/windows', function (req, res, next) {
    res.render('tools/windows');
});

router.get('/ubuntu', function (req, res, next) {
    res.render('tools/ubuntu');
});

module.exports = router;