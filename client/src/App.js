import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Homepage from "./pages/Homepage";
import Register from "./pages/Register";
import Login from "./pages/Login";
import { useSelector } from "react-redux";
import Spinner from "./components/Spinner";
import ProtectedRoute from "./components/protectedRoute";
import PublicRoute from "./components/PublicRoute";
import ApplyDoctor from "./pages/ApplyDoctor";
import NotificationPage from "./pages/NotificationPage";
import Users from "./pages/admin/Users";
import Doctors from "./pages/admin/Doctors";
import Profile from "./pages/doctor/Profile";
import BookingPage from "./pages/BookingPage";
import Appointment from "./pages/Appointment";
import DoctorAppointments from "./pages/doctor/DoctorAppointments";
import UserProfile from "./pages/user/UserProfile";
import HeartDiseasePrediction from "./algorithm/Heart_Disease_Prediction";
import IPage from "./components/IPage";
import ContactPage from "./components/ContactPage";
import OtherDoctors from "./components/OtherDoctors";
import Cardiologist from "./pages/doctor/Cardiologist";
import Optician from "./pages/doctor/Optician";
import Medicine from "./pages/doctor/Medicine";
import Orthopedic from "./pages/doctor/Orthopedic";
import Gynecologist from "./pages/doctor/Gynecologist";
import Dentist from "./pages/doctor/Dentist";
import Result from "./algorithm/Result";
import BookingProcess from "./pages/BookingProcess";


function App() {
  const { loading } = useSelector((state) => state.alerts);
  return (
    <>
      <BrowserRouter>
        {loading ? (
          <Spinner />
        ) : (
          <Routes>
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Homepage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/doctorList"
              element={
                <ProtectedRoute>
                  <OtherDoctors />
                </ProtectedRoute>
              }
            />
            <Route
              path="/index"
              element={
                <PublicRoute>
                  <IPage />
                </PublicRoute>
              }
            />
            <Route
              path="/apply-doctor"
              element={
                <ProtectedRoute>
                  <ApplyDoctor />
                </ProtectedRoute>
              }
            />
             <Route
              path="/cardiologist"
              element={
                <PublicRoute>
                  <Cardiologist />
                </PublicRoute>
              }
            />
             <Route
              path="/optician"
              element={
                <PublicRoute>
                  <Optician />
                </PublicRoute>
              }
            />
             <Route
              path="/medicine"
              element={
                <PublicRoute>
                  <Medicine />
                </PublicRoute>
              }
            />
             <Route
              path="/orthopedic"
              element={
                <PublicRoute>
                  <Orthopedic />
                </PublicRoute>
              }
            />
             <Route
              path="/gynecologist"
              element={
                <PublicRoute>
                  <Gynecologist />
                </PublicRoute>
              }
            />
             <Route
              path="/dentist"
              element={
                <PublicRoute>
                  <Dentist />
                </PublicRoute>
              }
            />
            <Route
              path="/notification"
              element={
                <ProtectedRoute>
                  <NotificationPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/contactpage"
              element={
                <PublicRoute>
                  <ContactPage />
                </PublicRoute>
              }
            />
            <Route
              path="/admin/users"
              element={
                <ProtectedRoute>
                  <Users />
                </ProtectedRoute>
              }
            />
            <Route
              path="/doctor/profile/:id"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/user/profile/:id"
              element={
                <ProtectedRoute>
                  <UserProfile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/appointments"
              element={
                <ProtectedRoute>
                  <Appointment />
                </ProtectedRoute>
              }
            />
            <Route
              path="/predict-heartDisease"
              element={
                <ProtectedRoute>
                  <HeartDiseasePrediction/>
                </ProtectedRoute>
              }
            />
             <Route
              path="/predict-result"
              element={
                <ProtectedRoute>
                  < Result/>
                </ProtectedRoute>
              }
            />
            <Route
              path="/doctor-appointments"
              element={
                <ProtectedRoute>
                  <DoctorAppointments />
                </ProtectedRoute>
              }
            />
            <Route
              path="/doctor/book-appointment/:doctorId"
              element={
                <ProtectedRoute>
                  {/* <BookingPage /> */}
                  <BookingProcess/>
                </ProtectedRoute>
              }
            />
            <Route
              path="/admin/doctors"
              element={
                <ProtectedRoute>
                  <Doctors />
                </ProtectedRoute>
              }
            />
            <Route
              path="/login"
              element={
                <PublicRoute>
                  <Login />
                </PublicRoute>
              }
            />
            <Route
              path="/register"
              element={
                <PublicRoute>
                  <Register />
                </PublicRoute>
              }
            />
          </Routes>
        )}
      </BrowserRouter>
    </>
  );
}

export default App;
