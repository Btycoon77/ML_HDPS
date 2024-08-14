import React, { useEffect, useState } from "react";
import Layout from "../components/Layout";
import axios from "axios";
import { Table, message } from "antd";
import moment from "moment";

const Appointment = () => {
  const [appointment, setappointment] = useState([]);

  const handleCancel = async (text, record) => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/cancel-appointment",
        {
          doctorId: record.doctorId,
          userId: record.userId,
        
        },
        {
          headers: {
            Authorization: "Bearer " + localStorage.getItem("token"),
          },
        }
      );
      if (res.data.success) {
        message.success(res.data.message);
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      console.log(error);
      message.error("Something went wrong!!");
    }
  };

  const getAppointments = async () => {
    try {
      const res = await axios.get("http://localhost:8000/api/v1/user/user-appointments", {
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

  const columns = [
    {
      title: "ID",
      dataIndex: "_id",
    },

    // {
    //   title: "Phone",
    //   dataIndex: "Phone",
    //   render: (text, record) => <span>{record.doctorId.phone}</span>,
    // },
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
      title: "Status",
      dataIndex: "status",
      render: (text, record) => <span>{record.status}</span>,
    },
    {
      title: "Action",
      dataIndex: "action",
      render: (text, record) => (
        <span>
          <button
            className="btn btn-danger"
            onClick={() => {
              handleCancel(text, record);
            }}
          >
            Cancel
          </button>
        </span>
      ),
    },
  ];

  return (
    <Layout>
      <h1 className="text-center">Appointment list</h1>
      <Table columns={columns} dataSource={appointment} />
    </Layout>
  );
};

export default Appointment;
