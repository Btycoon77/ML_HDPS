import React from "react";
import { adminMenu, userMenu } from "../data/SidebarMenu";
import "../styles/layout.css";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { message } from "antd";
import { Badge } from "antd";

const Layout = ({ children }) => {
  const location = useLocation();
  const { user } = useSelector((state) => state.user);
  const navigate = useNavigate();

  // ************doctor menu ********
  const doctorMenu = [
    {
      name: "Home",
      path: "/",
      icon: "fa-solid fa-house",
    },
    {
      name: "Appointments",
      path: "/doctor-appointments",
      icon: "fa-solid fa-list",
    },
    {
      name: "Profile",
      path: `/doctor/profile/${user?._id}`,
      icon: "fa-solid fa-user",
    },
  ];

   const userMenu = [
    {
        name:'Home',
        path:'/',
        icon:"fa-solid fa-house"
    },
    {
        name:'Appointments',
        path:'/appointments',
        icon:'fa-solid fa-list'
    },
    {
        name:'Apply doctor',
        path:'/apply-doctor',
        icon:'fa-solid fa-user-doctor'
    },
    {
        name:'profile',
        path:`/user/profile/${user?._id}`,
        icon:"fa-solid fa-user"
    },
    {
      name:'Doctor List',
      path:'/doctorList',
      icon:"fa-solid fa-user-doctor"
    },
    {
      name:'Predict Heart Disease',
      path:'/predict-heartDisease',
      icon:'fa-solid fa-user-doctor'
  },
  
];

  // ************doctor menu ********

  //    rendering menu list
  const sidebarMenu = user?.isAdmin
    ? adminMenu
    : user?.isDoctor
    ? doctorMenu
    : userMenu;

  //   logut function
  const handleLogout = () => {
    localStorage.clear();
    message.success("Logout successfully");
    navigate("/login");
  };

  return (
    <div className="main">
      <div className="layout">
        <div className="sidebar">
          <div className="logo">
            <h6>HDPS</h6>
            <hr />
          </div>
          <div className="menu">
            {sidebarMenu.map((data) => {
              const isActive = location.pathname === data.path;
              return (
                <>
                  <div className={`menu-item ${isActive && "active"}`}>
                    <i className={data.icon}></i>
                    <Link to={data.path}>{data.name}</Link>
                  </div>
                </>
              );
            })}

            {/* logout */}

            <div className={`menu-item `} onClick={handleLogout}>
              <i className={"fa-solid fa-right-from-bracket"}></i>
              <Link to={"/login"}>Logout</Link>
            </div>
          </div>
        </div>
        <div className="content">
          <div className="header">
            <div
              className="header-content"
              style={{
                cursor: "pointer",
              }}
            >
              <Badge
                count={user?.notification?.length}
                onClick={() => {
                  navigate("/notification");
                }}
              >
                <i class="fa-sharp fa-solid fa-bell"></i>
              </Badge>

              <Link to={"/profile"}>{user?.name}</Link>
            </div>
          </div>
          <div className="body">{children}</div>
        </div>
      </div>
    </div>
  );
};

export default Layout;
