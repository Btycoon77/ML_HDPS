const express = require('express');
const { getAllUserController, getAllDoctorController, changeAccountStatusController } = require('../controllers/adminController');
const adminRouter = express.Router();
const authMiddleware = require('../middlewares/authMiddleware')

//  get all users;

adminRouter.get('/getAllUsers',authMiddleware,getAllUserController);

//  GET METHOD || DOCTORS
adminRouter.get('/getAllDoctors',authMiddleware,getAllDoctorController);

// POST || ACCOUNT STATUS
adminRouter.post('/changeAccountStatus',authMiddleware,changeAccountStatusController);

module.exports = adminRouter;