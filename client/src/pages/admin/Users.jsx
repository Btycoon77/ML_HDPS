import { message, Table } from "antd";
import React, { useState, useEffect } from "react";
import Layout from "../../components/Layout";
import axios from "axios";
import {backend} from "../../utils/CONSTANTS"
const Users = () => {
  const [users, setusers] = useState([]);

  //  get users;
  const getusers = async () => {
    try {
      const res = await axios.get(`${backend}api/v1/admin/getAllUsers`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });
      if (res.data.success) {
        setusers(res.data.data);
      }
    } catch (error) {
      console.log(error);
      message.error("failed to get users");
    }
  };

  useEffect(() => {
    getusers();
  }, []);

  //  andD table col;
  const columns = [
    {
      title: "Name",
      dataIndex: "name",
    },
    {
      title: "Email",
      dataIndex: "email",
    },
    {
      title: "Doctor",
      dataIndex: "isDoctor",
      render: (text, record) => <span>{record.isDoctor ? "Yes" : "No"}</span>,
    },
    {
      title: "Actions",
      dataIndex: "actions",
      render: (text, record) => (
        <div className="d-flex">
          <button className="btn btn-danger">Block</button>
        </div>
      ),
    },
  ];

  return (
    <Layout>
      <h3 className="ms-3 text-center">Users list</h3>
      <Table columns={columns} dataSource={users}></Table>
    </Layout>
  );
};

export default Users;
