const appointmentModel = require("../models/appointmentModel");
const doctorModel = require("../models/doctorModels");
const userModel = require("../models/userModels");
const ObjectId = require("mongodb").ObjectId;

const getDoctorInfoController = async (req, res) => {
  try {
    const doctor = await doctorModel.findOne({ userId: req.body.userId });
    res.status(200).json({
      success: true,
      message: "doctor data fetch success",
      data: doctor,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "error in fetching doctor details",
    });
  }
};

// update doctor profile

const updateProfileController = async (req, res) => {
  try {
    const doctor = await doctorModel.findOneAndUpdate(
      { userId: req.body.userId },
      req.body
    );
    res.status(201).json({
      success: true,
      message: "Doctor profile updated",
      data: doctor,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "error in updating doctor profile",
    });
  }
};

//  get single doctor

const getDoctorByIdController = async (req, res) => {
  try {
    const doctor = await doctorModel.findOne({ _id: req.body.doctorId });
    res.status(200).json({
      success: true,
      message: "Single doctor info fetched",
      data: doctor,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "Error in getting single doctor",
      error,
    });
  }
};

const doctorAppointmentController = async (req, res) => {
  try {
    const doctor = await doctorModel.findOne({ userId: req.body.userId });
    const appointment = await appointmentModel.find({ doctorId: doctor._id });
    // getting doctor from userId;
    res.status(200).json({
      success: true,
      message: "Doctor appointments fetched successfully",
      data: appointment,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      error,
      success: false,
      message: "Error in Doctor appointment",
    });
  }
};

const updateStatusController = async (req, res) => {
  try {
    const { doctorId, status } = req.body;
    const userAppointment = await appointmentModel.find({ doctorId: doctorId });
    console.log(userAppointment);

    //  getting the user who have booked appointment from array of appointments ;
    const getUser = userAppointment.reduce((result, currentUser) => {
      if (currentUser.doctorId == doctorId) {
        (async () => {
          //  updating the user's status inside the appointment model to approved 
          await appointmentModel.updateOne(
            { _id: new ObjectId(currentUser._id) },
            { $set: { status: "Approved" } }
          );
        })();
        return currentUser; 
      } else {
        return result;
      }
    }, {});
    // console.log("this is get user",getUser);
    // console.log(getUser.userId);
    const user = await userModel.findOne({ _id: getUser.userId });
    // console.log(user);
    const notification = user.notification;
    notification.push({
      type: "Status updated",
      message: `your appointment has been ${status}`,
      onClickPath: "/doctor-appointments",
    });
    res.status(200).json({
      success: true,
      data: userAppointment,
      message: "Appointment status updated",
    });
    await user.save();
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      error,
      message: "Error in updating status",
    });
  }
};

module.exports = {
  getDoctorInfoController,
  updateProfileController,
  getDoctorByIdController,
  doctorAppointmentController,
  updateStatusController,
};
