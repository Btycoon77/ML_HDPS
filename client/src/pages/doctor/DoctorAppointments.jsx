import React, { useEffect, useState } from "react";
import Layout from "../../components/Layout";
import axios from "axios";
import { Table, message } from "antd";
import moment from "moment";

const DoctorAppointments = () => {
  const [appointment, setappointment] = useState([]);

  const getAppointments = async () => {
    try {
      const res = await axios.get("http://localhost:8000/api/v1/doctor/doctor-appointments", {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });
      if (res.data.success) {
        setappointment(res.data.data);
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      console.log(error);
    }
  };
  useEffect(() => {
    getAppointments();
  }, []);

  const handleStatus = async (record, status) => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/doctor/update-status",
        { doctorId: record.doctorId, status, userId: record.userId },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      if (res.data.success) {
        message.success(res.data.message);
        getAppointments();
      } else {
        message.error(res.data.error);
      }
    } catch (error) {
      console.log(error);
      message.error("something went wrong");
    }
  };

  const columns = [
    {
      title: "ID",
      dataIndex: "_id",
    },

    {
      title: "Date and Time",
      dataIndex: "date",
      render: (text, record) => (
        <span>
          {moment(record.date).format("DD-MM-YY")} {"  "}
          {moment(record.time).format("HH:mm")}
        </span>
      ),
    },
    {
      title: "Action",
      dataIndex: "action",
      render: (text, record) => (
        <div className="d-flex">
          {record.status == "pending" ? (
            <button
              className="btn btn-success"
              onClick={() => {
                handleStatus(record, "approved");
              }}
            >
              Approve
            </button>
          ) : (
            <button
              className="btn btn-danger ms-2"
              onClick={() => {
                handleStatus(record, "pending");
              }}
            >
              Reject
            </button>
          )}
        </div>
      ),
    },
  ];
  return (
    <Layout>
      <h1>DoctorAppointments</h1>
      <Table columns={columns} dataSource={appointment} />
    </Layout>
  );
};

export default DoctorAppointments;
