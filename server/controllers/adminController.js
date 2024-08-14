const userModel = require("../models/userModels");
const doctorModel = require("../models/doctorModels");

const getAllUserController = async (req, res) => {
  try {
    const user = await userModel.find();
    res.status(200).json({
      success: true,
      message: "users data list",
      data: user,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "failed to get all users",
      error: error,
    });
  }
};

const getAllDoctorController = async (req, res) => {
  try {
    const doctors = await doctorModel.find();
    res.status(200).json({
      success: true,
      message: "doctors data list",
      data: doctors,
    });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      success: false,
      message: "failed to get all doctors",
      error: error,
    });
  }
};

const changeAccountStatusController = async (req, res) => {
    try {
        const {doctorId,status} = req.body;
        const doctor =await doctorModel.findByIdAndUpdate(doctorId,{status},{new:true});
        const user = await userModel.findOne({_id:doctor.userId});
        const notification = user.notification;
        notification.push({
            type:'doctor-account-request-updated',
            message:`Your Doctor account request is ${status}`,
            onClickPath:'/notification'
        });
        const pdoctor = await doctorModel.findOne({_id:doctorId});
        console.log(pdoctor);
        pdoctor.status ==='Approved'?user.isDoctor=true:user.isDoctor=false;

        // this can be rewritten as
        // user.isDoctor = user.status ==='approved'?true:false;

        await user.save();
        res.status(201).json({
            success:true,
            message:'Account status updated',
            data:doctor
        })
    } catch (error) {
        console.log(error);
        res.status(500).json({
            success:false,
            message:'error in the account status',
            error
        });
    }
};

module.exports = {
  getAllDoctorController,
  getAllUserController,
  changeAccountStatusController,
};
