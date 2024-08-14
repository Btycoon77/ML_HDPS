import React from "react";
import { Button, message } from "antd";
import Form from "antd/es/form/Form";
import FormItem from "antd/es/form/FormItem";
import Input from "antd/es/input/Input";
import "../styles/register.css";
import { Link, useNavigate } from "react-router-dom";
import axios from 'axios';
import { useDispatch } from "react-redux";
import { hideLoading, showLoading } from "../redux/features/alertSlice";

const Register = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const onFinishHandler =async (values) => {
     try {
      dispatch(showLoading());
      const res = await axios.post('http://localhost:8000/api/v1/user/register',values);
      if(res.data.success){
        message.success('Registration succesfully done');
        navigate('/login');
        dispatch(hideLoading());
        
      }else{
        message.error('something went wrong');
      }
     } catch (error) {
      console.log(error);
      message.error('something went wrong');
     }
  };

  return (
    <>
      <div className="form-container">
        <Form
          layout="vertical"
          onFinish={onFinishHandler}
          className="register-form"
        >
          <h1 className="text-center">Register Form</h1>
          <FormItem label="Name" name="name" rules={[

            {required:true,message:"please enter your name"}
          ]

          }>
            <Input type="text" required 
            
            />
          </FormItem>

          <FormItem label="Email" name="email" rules={[
            {required:true,message:"please enter your email"}
          ]}>
            <Input type="email" required />
          </FormItem>

          <FormItem label="Password" name="password" rules={[
            {required:true,message:"please enter strong password"}
          ]}>
            <Input type="password" required />
          </FormItem>

          <Link to={"/login"} className="m-2">Already user,login here</Link>

          <button type="submit" className="btn btn-primary">
            Register
          </button>
        </Form>
      </div>
    </>
  );
};

export default Register;
