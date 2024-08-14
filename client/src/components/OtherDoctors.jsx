import { Row } from "antd";
import Layout from "../components/Layout";
import axios from "axios";
import React, { useEffect, useState } from "react";
import DoctorList from "../components/DoctorList";
import { useSelector } from "react-redux";

const OtherDoctors = () => {
  const { user } = useSelector((state) => state.user);
  const [doctors, setDoctors] = useState([]);
  const getUserData = async () => {
    try {
      const res = await axios.get(
        "http://localhost:8000/api/v1/user/getAllDoctors",

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
        <h1 className="text-center">Doctor List</h1>
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

export default OtherDoctors;
