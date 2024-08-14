import React, { useEffect, useState } from "react";
import Layout from "../../components/Layout";
import { message, Table } from "antd";
import axios from "axios";

const Orthopedic = () => {
  const [doctors, setDoctors] = useState([]);

  //  get all doctors list
  const getDoctors = async () => {
    try {
      const res = await axios.get(
        "http://localhost:8000/api/v1/user/getAllOrthopedic",
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      if (res.data.success) {
        setDoctors(res.data.data);
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      console.log(error);
      message.error("Failed to get all doctor list");
    }
  };
  useEffect(() => {
    getDoctors();
  }, []);

  //  columns for doctor;

  const col = [
    {
      title: "Name",
      dataIndex: "name",
      render: (text, record) => (
        <span>
          {record.firstName + " "}
          {record.lastName}
        </span>
      ),
    },
    {
      title: "Email",
      dataIndex: "email",
    },
    {
      title: "Specialization",
      dataIndex: "specialization",
    },
    {
      title: "Consultation Fees",
      dataIndex: "feesPerConsultation",
    },
  ];

  return (
    <>
      <h1 className="text-center ">Orthopedic list</h1>
      <div className="container mx-2 my-2">
        <Table dataSource={doctors} columns={col} bordered={true}></Table>
      </div>
    </>
  );
};

export default Orthopedic;
