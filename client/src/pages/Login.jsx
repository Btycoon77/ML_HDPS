import { Form, Input, message } from "antd";
import FormItem from "antd/es/form/FormItem";
import axios from "axios";
import React from "react";
import { useDispatch } from 'react-redux';
import { showLoading,hideLoading } from "../redux/features/alertSlice";
import { Link, useNavigate } from "react-router-dom";
import "../styles/register.css";

const Login = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const onFinishHandler = async (values) => {
    try {
      dispatch(showLoading());
      const res = await axios.post("http://localhost:8000/api/v1/user/login", values, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      window.location.reload();
      dispatch(hideLoading());
      if (res.data.success) {
        localStorage.setItem('token',res.data.token);
        message.success("Login successfull");
        navigate('/');
      }else{
        message.error('user not found');
      }
    } catch (error) {
      console.log(error);
      message.error("something went wrong");
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
          <h1 className="text-center">Login Form</h1>

          <FormItem label="Email" name="email">
            <Input type="email" required />
          </FormItem>

          <FormItem label="Password" name="password">
            <Input type="password" required />
          </FormItem>

          <Link to={"/register"} className="m-2">
            Not a user,register here
          </Link>

          <button type="submit" className="btn btn-primary">
            Login
          </button>
        </Form>
      </div>
    </>
  );
};

export default Login;
