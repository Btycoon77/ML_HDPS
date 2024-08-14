const userModel = require("../models/userModels");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const doctorModel = require("../models/doctorModels");
const appointmentModel = require("../models/appointmentModel");
const moment = require("moment");
const userSchema = require("../validator/userValidation");
const ObjectId = require("mongodb").ObjectId;
const path = require("path");
const registerController = async (req, res) => {
  try {
    const existing_user = await userModel.findOne({ email: req.body.email });

    if (existing_user) {
      return res
        .status(400)
        .json({ message: "user already exist", success: false });
    }
    const password = req.body.password;
    const salt = await bcrypt.genSalt(10);
    const hashed_password = await bcrypt.hash(password, salt);
    req.body.password = hashed_password;
    // const validatedUser = userSchema.validate(req.body);
    // await userSchema.validateAsync(req.body);
    const new_user = new userModel(req.body); // validatedUser
    await new_user.save();
    res.status(201).json({ message: "Registration succesfull", success: true });
  } catch (error) {
    console.log(error);
    res.status(500).json({ success: false, message: "Registration failed " });
  }
};

// login handler
const loginController = async (req, res) => {
  try {
    const user = await userModel.findOne({ email: req.body.email });
    if (!user) {
      return res
        .status(200)
        .json({ message: "user not found", success: false });
    }
    const isMatched = await bcrypt.compare(req.body.password, user.password);
    if (!isMatched) {
      return res.status(400).json({
        message: "Invalid email or password",
        success: false,
      });
    }
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, {
      expiresIn: "1d",
    });
    res.status(200).json({
      message: "Login success",
      success: true,
      token,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: `Error in login is ${error.message}`,
    });
  }
};

const authController = async (req, res, next) => {
  try {
    //  if I am fetching the _id from the collection in database then you have to enclose it within ObjectId('id');
    // else not  you can do this way;
    //  if you are taking it from the user then you can do it this way
    const user = await userModel.findById({
      _id: req.body.userId,
    });
    const isAdmin = user.isAdmin ? true : false;
    user.password = undefined;

    if (!user) {
      return res.status(200).send({
        message: "user not found",
        success: false,
      });
    } else {
      res.status(200).send({
        success: true,
        data: user,
      });
    }
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Auth error",
      error,
    });
  }
};

// apply doctor controller
const applyDoctorController = async (req, res) => {
  try {
    const newDoctor = await doctorModel({ ...req.body, status: "pending" });
    await newDoctor.save();
    const adminUser = await userModel.findOne({ isAdmin: true });
    const notification = adminUser.notification;
    notification.push({
      type: "apply-doctor-request",
      message: `${newDoctor.firstName} ${newDoctor.lastName} has applied for a doctor account`,
      data: {
        doctorId: newDoctor._id,
        name: newDoctor.firstName + " " + newDoctor.lastName,
        onClickPath: "/admin/doctors",
      },
    });
    await userModel.findByIdAndUpdate(adminUser._id, { notification });
    res.status(201).json({
      success: true,
      message: "Doctor account applied succesfully",
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "Error while applying for doctor",
    });
  }
};

//  notification controller

const getAllNotificationController = async (req, res) => {
  try {
    const user = await userModel.findOne({ _id: req.body.userId });
    const seeNotification = user.seeNotification;
    const notification = user.notification;
    seeNotification.push(...notification);
    user.notification = [];
    user.seeNotification = notification;
    const updateUser = await user.save();
    res.status(200).json({
      success: true,
      message: "all notification marked as read",
      data: updateUser,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Error in notification",
    });
  }
};

//  delete notifications

const deleteAllNotificationController = async (req, res) => {
  try {
    const user = await userModel.findOne({ _id: req.body.userId });
    user.notification = [];
    user.seeNotification = [];
    const updatedUser = await user.save();
    updatedUser.password = undefined;
    res.status(200).json({
      success: true,
      message: "notifications deleted successfully",
      data: updatedUser,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      sucess: false,
      message: "unable to delete all message",
      error,
    });
  }
};

// get all doctor controller

const getAllDoctorController = async (req, res) => {
  try {
    const doctors = await doctorModel.find({ status: "Approved" });
    res.status(200).json({
      success: true,
      message: "Doctors list fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching doctor ",
    });
  }
};

//  get only cardiologist

const getAllCardiologist = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "Cardiologist",
    });
    res.status(200).json({
      success: true,
      message: "Cardiologist fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching cardiologist",
    });
  }
};

//  get only opticain

const getAllOptician = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "optician",
    });
    res.status(200).json({
      success: true,
      message: "Optician fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching optician",
    });
  }
};

//  get only Medicine

const getAllMedicine = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "Medicine",
    });
    res.status(200).json({
      success: true,
      message: "Medicine fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching Medicine doctor",
    });
  }
};

//  get only Orthopedic doctors

const getAllOrthopedic = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "Orthopedic",
    });
    res.status(200).json({
      success: true,
      message: "Orthopedic fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching orthopedic doctor",
    });
  }
};

//  get only Gynecologist

const getAllGynecologist = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "Gynecologist",
    });
    res.status(200).json({
      success: true,
      message: "Gynecologist fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching Gynecologist doctor",
    });
  }
};

//  get only Dentist

const getAllDentist = async (req,res) => {
  try {
    const doctors = await doctorModel.find({
      status: "Approved",
      specialization: "dentist",
    });
    res.status(200).json({
      success: true,
      message: "Dentist fetched succesfully",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error: error,
      message: "Error while fetching Dentist doctor",
    });
  }
};

//  book appointment

const bookAppointmentController = async (req, res) => {
  try {
    req.body.date = moment(req.body.date, "DD-MM-YYYY").toISOString();
    req.body.time = moment(req.body.time, "HH:mm");
    req.body.status = "pending";
    const newAppointment = new appointmentModel(req.body);
    await newAppointment.save();
    const user = await userModel.findById({ _id: req.body.doctorInfo.userId });
    user.notification.push({
      type: "New-appointment-request",
      message: `A new appointment request from ${req.body.userInfo.name} `,
      onClickPath: "/user/appointments",
    });
    await user.save();
    res.status(200).json({
      success: true,
      message: "appointment book succesfully",
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Error in booking appointment",
      error,
    });
  }
};

//  booking availability controller;

const bookingAvailabilityController = async (req, res) => {
  try {
    const date = moment(req.body.date, "DD-MM-YY").toISOString();
    const fromTime = moment(req.body.time, "HH:mm")
      .subtract(1, "hours")
      .toISOString();
    const toTime = moment(req.body.time, "HH:mm").add(1, "hours").toISOString();
    const doctorId = req.body.doctorId;
    const appointments = await appointmentModel.find({
      doctorId,
      date,
      time: {
        $gte: fromTime,
        $lte: toTime,
      },
    });

    if (appointments.length > 0) {
      return res.status(200).json({
        message: "Appointment not available after this",
        success: true,
      });
    } else {
      res.status(200).json({
        success: true,
        message: "Appointments available",
      });
    }
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Error in booking ",
      error,
    });
  }
};

//  appointment list controller

const userAppointmentController = async (req, res) => {
  try {
    const appointment = await appointmentModel.find({
      userId: req.body.userId,
    });
    res.status(200).json({
      success: true,
      message: "User appointmetns fetched succesfully",
      data: appointment,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Error in getting appointment",
      error,
    });
  }
};

const getUserInfoController = async (req, res) => {
  try {
    const user = await userModel.findOne({ _id: req.body.userId });
    res.status(200).json({
      success: true,
      message: "user data fetch success",
      data: user,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "error in fetching user details",
    });
  }
};

// update doctor profile

const updateProfileController = async (req, res) => {
  try {
    const user = await userModel.findOneAndUpdate(
      { _id: req.body.userId },
      req.body
    );
    res.status(201).json({
      success: true,
      message: "User profile updated",
      data: user,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "error in updating user profile",
    });
  }
};

//  cancel appointment controller

const cancelAppointmentController = async (req, res) => {
  try {
    const doctorId = req.body.doctorId;
    const userId = req.body.userId;

    const appointment = await appointmentModel.findOne({
      doctorId: doctorId,
      userId: userId,
    });

    const deleteAppointment = await appointmentModel.deleteOne({
      _id: appointment._id,
    });
    // await appointment.save();

    const doctor = await doctorModel.findOne({ _id: appointment.doctorId });
    console.log("this is doctor", doctor);

    const notification = doctor.notification;
    console.log("this is notification of doctor", notification);
    notification.push({
      type: "cancel-appointment",
      message: "Appointment cancelled succesfully",
      clickPath: "/cancel-appointment",
    });
    await doctor.save();
    res.status(200).json({
      success: true,
      message: "Appointment cancelled succesfully",
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      error,
      message: "Cannot cancel appointment",
      success: false,
    });
  }
};

const getModel = async (req, res) => {
  try {
    const modelDir = path.join(__dirname, "models");
    const modelPath = path.join(modelDir, "model.json");
    console.log(modelPath);
    res.sendFile(modelPath);
  } catch (error) {
    console.log(error);
    res.status(500).json("could not get the model");
  }
};

module.exports = {
  applyDoctorController,
  loginController,
  registerController,
  authController,
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
  getAllDentist,
  getAllGynecologist,
  getAllMedicine,
  getAllOptician,
  getAllOrthopedic
};
