import React, { useEffect, useState } from "react";
import Layout from "../../components/Layout";
import axios from "axios";
import { TimePicker, message } from "antd";
import { useNavigate, useParams } from "react-router-dom";
import { Col, Form, Input, Row } from "antd";
import { useDispatch, useSelector } from "react-redux";
import moment from "moment";
import { hideLoading, showLoading } from "../../redux/features/alertSlice";

const UserProfile = () => {
  const [users, setUsers] = useState([]);

  const navigate = useNavigate();
  const dispatch = useDispatch();
  const params = useParams();

  const { user } = useSelector((state) => state.user);

  // update user and handling form

  const handleForm = async (values) => {
    try {
      console.log(values);
      dispatch(showLoading());
      const res = await axios.put(
        "http://localhost:8000/api/v1/user/updateProfile",
        {
          ...values,
          userId: user._id,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );

      dispatch(hideLoading());
      if (res.data.success) {
        message.success(res.data.message);
        console.log(values);
        navigate("/");
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      console.log(error);
      message.error("Something went wrong");
    }
  };

  //  get user info

  const getUserInfo = async () => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/getUserInfo",
        {
          userId: params.id,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      if (res.data.success) {
        setUsers(res.data.data);
        message.success(res.data.message);
      } else {
        message.error(res.data.message);
      }
      console.log(user);
    } catch (error) {
      message.error("something went wrong");
    }
  };

  useEffect(() => {
    getUserInfo();
  }, []);

  return (
    <Layout>
      <h1>Manage Profile</h1>
      {users && (
        <Form
          layout="vertical"
          onFinish={handleForm}
          className="m-3"
          initialValues={users}
        >
          <Row gutter={20}>
            <Col xs={24} md={24} lg={8}>
              <h4>Personal details:</h4>
              <Form.Item
                label="First name"
                name="firstName"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your first name" />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Last name"
                name="lastName"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your last name" />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="phone no"
                name="phone"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your phone no" />
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Website"
                name="website"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your website" />
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Address"
                name="address"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your address" />
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Email"
                name="email"
                required
                rules={[{ required: true }]}
              >
                <Input type="text" placeholder="your email" />
              </Form.Item>
            </Col>
          </Row>

          {/*  professional details */}
          {user?.isDoctor ? (
            <Row gutter={20}>
              <Col xs={24} md={24} lg={8}>
                <h4>Professional details:</h4>
                <Form.Item
                  label="Specialization"
                  name="specialization"
                  required
                  rules={[{ required: true }]}
                >
                  <Input type="text" placeholder="your specialization" />
                </Form.Item>
              </Col>
              <Col xs={24} md={24} lg={8}>
                <Form.Item
                  label="Experience "
                  name="experience"
                  required
                  rules={[{ required: true }]}
                >
                  <Input type="text" placeholder="your experience" />
                </Form.Item>
              </Col>
              <Col xs={24} md={24} lg={8}>
                <Form.Item
                  label="Fees Per Consultation"
                  name="feesPerConsultation"
                  required
                  rules={[{ required: true }]}
                >
                  <Input type="text" placeholder="your fee consulation" />
                </Form.Item>
              </Col>

              <Col xs={24} md={24} lg={8}>
                <Form.Item
                  label="Timings"
                  name="timings"
                  required
                  rules={[{ required: true }]}
                >
                  <TimePicker.RangePicker />
                  {/* <Input
                  type="text"
                  placeholder={[
                    moment(user.timings[0]).format("HH:mm"),
                    moment(user.timings[1]).format("HH:mm"),
                  ]}
                /> */}
                </Form.Item>
              </Col>
            </Row>
          ) : (
            <></>
          )}
          <div className="d-flex justify-content-end">
            <button className="btn btn-primary form-btn" type="submit">
              Update
            </button>
          </div>
        </Form>
      )}
    </Layout>
  );
};

export default UserProfile;
