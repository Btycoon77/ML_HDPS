const express = require("express");
const {
  loginController,
  registerController,
  authController,
  applyDoctorController,
  getAllNotificationController,
  deleteAllNotificationController,
  getAllDoctorController,
  bookAppointmentController,
  bookingAvailabilityController,
  userAppointmentController,
  getUserInfoController,
  updateProfileController,
  cancelAppointmentController,
  getModel,
  getAllCardiologist,
  getAllMedicine,
  getAllOptician,
  getAllOrthopedic,
  getAllGynecologist,
  getAllDentist,
} = require("../controllers/userController");
const authMiddlware = require("../middlewares/authMiddleware");

const predictHeartDiseaseController = require("../controllers/HeartDiseasePredictionController");
const HeartDiseasePredictionController = require("../controllers/HeartPredictionController");

const userRouter = express.Router();

//  Login || post
userRouter.post("/login", loginController);

// Registration || post
userRouter.post("/register", registerController);

//  Auth ||post
userRouter.post("/getUserData", authMiddlware, authController);

// apply Doctor
userRouter.post("/apply-doctor", authMiddlware, applyDoctorController);

//  notification Doctor  || POST

userRouter.post(
  "/get-all-notification",
  authMiddlware,
  getAllNotificationController
);

//  notification Doctor  || POST

userRouter.post(
  "/delete-all-notification",
  authMiddlware,
  deleteAllNotificationController
);

//  get all doctor

userRouter.get("/getAllDoctors", authMiddlware, getAllDoctorController);

//  get all specific  doctors;

userRouter.get("/getAllCardiologist", getAllCardiologist);
userRouter.get("/getAllOpticain", getAllOptician);
userRouter.get("/getAllMedicine", getAllMedicine);
userRouter.get("/getAllOrthopedic",  getAllOrthopedic);
userRouter.get("/getAllGynecologist", getAllGynecologist);
userRouter.get("/getAllDentist",  getAllDentist);

//  book appointment

userRouter.post("/book-appointment", authMiddlware, bookAppointmentController);

//  booking availabiility
userRouter.post(
  "/booking-availability",
  authMiddlware,
  bookingAvailabilityController
);

//  appointment list

userRouter.get("/user-appointments", authMiddlware, userAppointmentController);

//  cancelling the appointment by user.

userRouter.post(
  "/cancel-appointment",
  authMiddlware,
  cancelAppointmentController
);

userRouter.post("/getUserInfo", authMiddlware, getUserInfoController);

//  POST || UPDATE PROFILE

userRouter.put("/updateProfile", authMiddlware, updateProfileController);

//  post || GET SINGLE DOCTOR INFO

// POST || predict heart disease

// userRouter.post('/predict',HeartDiseasePredictionController);

//  loading the model

// userRouter.get('/model',getModel);

module.exports = userRouter;
