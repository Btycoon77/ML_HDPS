import Layout from "../components/Layout";
import React from "react";
import { message, Tabs } from "antd";
import TabPane from "antd/es/tabs/TabPane";
import { useDispatch, useSelector } from "react-redux";
import { hideLoading, showLoading } from "../redux/features/alertSlice";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const NotificationPage = () => {
  const { user } = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  //   handle read notification;
  const handleMarkAllRead = async () => {
    try {
      dispatch(showLoading());
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/get-all-notification",
        { userId: user._id },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      dispatch(hideLoading());
      if (res.data.success) {
        message.success(res.data.message);
        window.location.reload();
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      dispatch(hideLoading());
      console.log(error);
      message.error("something went wrong");
    }
  };

  const handleDeleteAllRead = async () => {
    try {
      dispatch(showLoading());
      const res = await axios.post(
        "http://localhost:8000/api/v1/user/delete-all-notification",
        { userId: user._id },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      dispatch(hideLoading());
      if (res.data.success) {
        message.success(res.data.message);
        window.location.reload();
      } else {
        message.error(res.data.message);
      }
    } catch (error) {
      console.log(error);
      message.error("something went wrong");
    }
  };

  return (
    <>
      <Layout>
        <h4 className="p-3 text-center">NotificationPage</h4>
        <Tabs>
          <TabPane tab="unRead" key={0}>
            <div className="d-flex justify-content-end">
              <h4
                className="p-2 "
                style={{
                  cursor: "pointer",
                }}
                onClick={handleMarkAllRead}
              >
                Mark all Read
              </h4>
            </div>
            {user?.notification.map((not) => {
              return (
                <>
                  <div className="card" style={{ cursor: "pointer" }}>
                    <div
                      className="card-text"
                      onClick={() => {
                        navigate(not.onClickPath);
                      }}
                    >
                      {not.message}
                    </div>
                  </div>
                </>
              );
            })}
          </TabPane>

          <TabPane tab="Read" key={1}>
            <div className="d-flex justify-content-end">
              <h4
                className="p-2 text-primary"
                style={{ cursor: "pointer" }}
                onClick={handleDeleteAllRead}
              >
                Delete all Read
              </h4>
            </div>
            {user?.seeNotification.map((not) => {
              return (
                <>
                  <div className="card" style={{ cursor: "pointer" }}>
                    <div
                      className="card-text"
                      onClick={() => {
                        navigate(not.onClickPath);
                      }}
                    >
                      {not.message}
                    </div>
                  </div>
                </>
              );
            })}
          </TabPane>
        </Tabs>
      </Layout>
    </>
  );
};

export default NotificationPage;
