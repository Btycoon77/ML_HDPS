import { Row } from "antd";
import Layout from "../components/Layout";
import axios from "axios";
import React, { useEffect, useState } from "react";
import DoctorList from "../components/DoctorList";
import {  useSelector } from "react-redux";


const Homepage = () => {
  
  const { user } = useSelector((state) => state.user);
  const [doctors, setDoctors] = useState([]);
  const getUserData = async () => {
    try {
      const res = await axios.get(
        "http://localhost:8000/api/v1/user/getAllCardiologist",

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

  useEffect(() => {
    getUserData();
  }, []);

  return (
    <>
      <Layout>
        <h2 className="text-center">Book Appointment With Cardiologist</h2>
        <Row>
          {doctors &&
            doctors.map((doctor) => {
              return (
                <>
               
                  <DoctorList doctor={doctor} />
                </>
              );
            })}
        </Row>
      </Layout>
    </>
  );
};

export default Homepage;
