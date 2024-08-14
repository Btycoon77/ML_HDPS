import React from "react";
import Layout from "../components/Layout";
import { Col, Form, Input, Row ,TimePicker} from "antd";
import {message} from 'antd';
import {useSelector,useDispatch} from 'react-redux';
import { useNavigate } from "react-router-dom";
import { hideLoading, showLoading } from "../redux/features/alertSlice";
import axios from "axios";
import moment from "moment";


const ApplyDoctor = () => {
  // handle form
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const {user} = useSelector((state)=>state.user);


  const handleForm = async(values) => {
    try {
        console.log(values);
        dispatch(showLoading());
        const res = await axios.post('http://localhost:8000/api/v1/user/apply-doctor',{...values,
        timings:[
          moment(values.timings[0]).format("HH:mm"),
          moment(values.timings[1]).format("HH:mm"),
        ]},{
            headers:{
                Authorization:`Bearer ${localStorage.getItem('token')}`
            }
        });
        dispatch(hideLoading());
        if(res.data.success){
            message.success('you will be notified by admin')
            navigate('/')
        }else{
            message.error('something went wrong');
        }
    } catch (error) {
        console.log(error);
        message.error('Something went wrong');
    }
  };
  return (
    <Layout>
      <h1 className="text-center">Apply doctor</h1>
      <Form layout="vertical" onFinish={handleForm} className="m-3">
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
              <Input type="email" placeholder="your email" />
            </Form.Item>
          </Col>
        </Row>

        {/*  professional details */}

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
              <TimePicker.RangePicker format="HH:mm"/>
            </Form.Item>
          </Col>
        </Row>
        <div className="d-flex justify-content-end">
            <button className="btn btn-primary form-btn">
                Submit
            </button>
        </div>
      </Form>
    </Layout>
  );
};

export default ApplyDoctor;
