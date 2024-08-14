import React, { useEffect, useState } from "react";
import Layout from "../../components/Layout";
import { message, Table } from "antd";
import axios from "axios";

const Doctors = () => {
  const [doctors, setDoctors] = useState([]);

  const handleAccountStatus = async (record, status) => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/admin/changeAccountStatus",
        {
          doctorId: record._id,
          userId: record.userId,
          status: status,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      if (res.data.success) {
        message.success(res.data.message);
        window.location.reload();
      }
    } catch (error) {
      message.error("something went wrong");
    }
  };

  //  get all doctors list
  const getDoctors = async () => {
    try {
      const res = await axios.get("http://localhost:8000/api/v1/admin/getAllDoctors", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });
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
      title: "Action",
      dataIndex: "actions",
      render: (text, record) => (
        <div className="flex">
          {record.status == "pending" ? (
            <button
              className="btn btn-success"
              onClick={() => {
                handleAccountStatus(record, "Approved");
              }}
            >
              Approve
            </button>
          ) : (
            <button className="btn btn-danger ms-2">Reject</button>
          )}
        </div>
      ),
    },
  ];

  return (
    <Layout>
      <h1 className="text-center ">Doctors list</h1>
      <Table dataSource={doctors} columns={col}></Table>
    </Layout>
  );
};

export default Doctors;
