const express = require("express");
const doctorRouter = express.Router();
const authMiddleware = require("../middlewares/authMiddleware");
const {
  getDoctorInfoController,
  updateProfileController,
  getDoctorByIdController,
  doctorAppointmentController,
  updateStatusController,
} = require("../controllers/doctorController");

//  post || SINGLE DOCTOR INFO

doctorRouter.post("/getDoctorInfo", authMiddleware, getDoctorInfoController);

//  POST || UPDATE PROFILE

doctorRouter.put("/updateProfile", authMiddleware, updateProfileController);

//  post || GET SINGLE DOCTOR INFO

doctorRouter.post("/getDoctorById", authMiddleware, getDoctorByIdController);

// GET  appointments

doctorRouter.get(
  "/doctor-appointments",
  authMiddleware,
  doctorAppointmentController
);

//  POST  updatae status

doctorRouter.post("/update-status", authMiddleware, updateStatusController);

module.exports = doctorRouter;
