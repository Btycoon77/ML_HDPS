import React from "react";
import "../styles/Ipage.css";
import SearchIcon from "../images/search.png";
import VoiceSearch from "../images/voice-search.png";
import Video from "../assets/d.mp4";
import Logo from "../images/logo1.png";

const IPage = () => {
  return (
    <>
      {/* <!-- -------------------start of navabar------------- --> */}
      <nav>
        <div className="logo">
          <img src={Logo} alt="logo" />
          <span>HDPS</span>
        </div>
        <ul>
          <li>
            <a href="/index">Home</a>
          </li>

          <li>
            <a href="#">
              Doctors
              <i
                style={{ marginLeft: "5px" }}
                className="fas fa-caret-down"
              ></i>
            </a>

            <div className="dropdown-menu">
              <ul>
                <li>
                  <a className="dropdown-item" href="/cardiologist">
                    Cardiologist
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/medicine">
                    Medicine
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/orthopedic">
                    Orthopedic
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/dentist">
                    Dentist
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/gynecologist">
                    Gynecologist
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/optician">
                    Optician
                  </a>
                </li>
              </ul>
            </div>
          </li>
          <li>
            <a href="/contactpage">Contact us</a>
          </li>

          <li>
            <a href="#">
              Login{" "}
              <i
                style={{ marginLeft: "5px" }}
                className="fas fa-caret-down"
              ></i>
            </a>

            <div className="dropdown-menu">
              <ul>
                <li>
                  <a className="dropdown-item" href="/login">
                    User
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="/login">
                    Doctor
                  </a>
                </li>

                <li>
                  <a className="dropdown-item" href="/login">
                    Admin
                  </a>
                </li>
              </ul>
            </div>
          </li>
        </ul>
        {/* <!-- ----------------searchbar-------------- --> */}
        <div className="nav-middle flex-div">
          <div className="search-box flex-div">
            <input type="text" name="" placeholder="Search" id="" />
            <img className="search-icon" src={SearchIcon} alt="search" />
          </div>
          <img className="mic-icon" src={VoiceSearch} alt="mic" />
        </div>
      </nav>
      {/* <!-- -------------------------end of navbar---------------- --> */}

      <div className="content">
        <div className="vid-parent">
          <video src={Video} playsInline autoPlay muted loop></video>
          <div className="vid-overlay"></div>
        </div>
        <div className="vid-content">
          <h1 className="slide-left">
            Welcome to <br />
            Heart Disease Prediction System
          </h1>
          <p className="slide-left">
            Our highly specialized experts are deeply experienced in treating
            rare and complex condtions.If you are not registered, please feel
            free to register.
          </p>
          <div className="links slide-left">
            <a href="/login" className="btn">
              Sign up
            </a>
            <a href="/contactpage">Contact us</a>
          </div>
        </div>
      </div>
    </>
  );
};

export default IPage;
