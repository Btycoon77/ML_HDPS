import React, { useEffect, useState } from "react";
import Layout from "../components/Layout";
import axios from "axios";
import { useParams } from "react-router-dom";
import { DatePicker, message, Card } from "antd";
import moment from "moment";
import { useDispatch, useSelector } from "react-redux";
import { showLoading, hideLoading } from "../redux/features/alertSlice";
import TimePicker from "react-time-picker";

const BookingPage = () => {
  const [doctors, setDoctors] = useState([]);
  const params = useParams();
  const [date, setDate] = useState();
  const [time, setTime] = useState();
  const [isAvailable, setisAvailable] = useState();
  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.user);

  // ************ login user dat***********

  const getUserData = async () => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/doctor/getDoctorById",
        { doctorId: params.doctorId },
        {
          headers: {
            Authorization: "Bearer " + localStorage.getItem("token"),
          },
        }
      );
      if (res.data.success) {
        setDoctors(res.data.data);
      }
    } catch (error) {
      console.log(error);
    }
  };

  // *********** Booking function*********

  const handleBooking = async () => {
    try {
      setisAvailable(true);
      if (!date && !time) {
        setisAvailable(false);
        console.log(isAvailable);
        return alert("Date and time required");
      } else {
        setisAvailable(true);
        console.log(isAvailable);
      }
      dispatch(showLoading());
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/book-appointment",
        {
          doctorId: params.doctorId,
          userId: user._id,
          doctorInfo: doctors,
          userInfo: user,
          date: date,
          time: time,
        },
        {
          headers: {
            Authorization: "Bearer " + localStorage.getItem("token"),
          },
        }
      );
      dispatch(hideLoading());
      if (res.data.success) {
        message.success(res.data.message);
      }
    } catch (error) {
      console.log(error);
    }
  };

  //  ************* booking availabliity*********

  const handleAvailability = async () => {
    try {
      dispatch(showLoading());
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/booking-availability",
        {
          doctorId: params.doctorId,
          date: date,
          time: time,
        },
        {
          headers: {
            Authorization: "Bearer " + localStorage.getItem("token"),
          },
        }
      );
      dispatch(hideLoading());
      if (res.data.success) {
        setisAvailable(true);
        message.success(res.data.message);
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      dispatch(hideLoading());
      console.log(error);
    }
  };

  useEffect(() => {
    getUserData();
    // eslint-disable--next-line
  }, []);

  return (
    <>
      <Card>
        <h3 className="text-center">Booking page</h3>
        <div className="container">
          {doctors && (
            <>
              <div>
                <h4>
                  Dr.{doctors.firstName} {doctors.lastName}
                </h4>
                <h4>Fees: Rs.{doctors.feesPerConsultation} </h4>
                {/* <h4>Timings:  Dr.{doctors?.timings[0]} - {doctors?.timings[1]} </h4> */}
                <div className="d-flex flex-column w-50">
                  <DatePicker
                    aria-required={"true"}
                    className="m-2"
                    format={"DD-MM-YY"}
                    onChange={(value) => {
                      setDate(moment(value).format("DD-MM-YY"));
                    }}
                  />

                  <TimePicker
                    aria-required={"true"}
                    // format="HH:mm"
                    className="mt-2"
                    onChange={(value) => {
                      setTime(value);
                      console.log(value);
                    }}
                    value={time}
                  />
                  {/* <button
                    className="btn btn-primary mt-2"
                    onClick={handleAvailability}
                  >
                    Check Availability
                  </button> */}

                  <button className="btn btn-dark mt-2" onClick={handleBooking}>
                    Book Now
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </Card>
    </>
  );
};

export default BookingPage;
