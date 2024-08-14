import React from "react";
// import '../styles/index.css';

const Navbar = () => {
  return (
    <>
      <nav>
        <div className="logo">
          <img src="images/logo1.png" alt="logo" />
          <span>e-Healthcare</span>
        </div>
        <ul>
          <li>
            <a href="index.jsx">Home</a>
          </li>

          <li>
            <a href="#">
              Doctors<i style="margin-left: 5px;" className="fas fa-caret-down"></i>
            </a>

            <div className="dropdown-menu">
              <ul>
                <li>
                  <a className="dropdown-item" href="cardiologist.jsx">
                    Cardiologist
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="medicine.jsx">
                    Medicine
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="orthopedic.jsx">
                    Orthopedic
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="surgeon.jsx">
                    Surgeon
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="gynecologist.jsx">
                    Gynecologist
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="orthopedic.jsx">
                    Orthopedic
                  </a>
                </li>
                <li>
                  <a
                    className="dropdown-item"
                    href="cardiac_electrophysiologist.jsx"
                  >
                    Cardiac electrophysiologist
                  </a>
                </li>
              </ul>
            </div>
          </li>
          <li>
            <a href="contactpage.jsx">Contact us</a>
          </li>

          <li>
            <a href="#">
              Login <i style="margin-left: 5px;" className="fas fa-caret-down"></i>
            </a>

            <div className="dropdown-menu">
              <ul>
                <li>
                  <a className="dropdown-item" href="users/login.jsx">
                    User
                  </a>
                </li>
                <li>
                  <a className="dropdown-item" href="doctors/doctor_login.jsx">
                    Doctor
                  </a>
                </li>

                <li>
                  <a className="dropdown-item" href="admin/admin_login.jsx">
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
            <img className="search-icon" src="images/search.png" alt="search" />
          </div>
          <img className="mic-icon" src="images/voice-search.png" alt="mic" />
        </div>
      </nav>
    </>
  );
};

export default Navbar;
