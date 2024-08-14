import React, { useContext, useEffect, useState } from "react";
import KhaltiCheckout from "khalti-checkout-web";
import { Button, Card } from "antd";
import { PaymentContext } from "../pages/PaymentContext";
import { useLocation } from "react-router-dom"; // To get doctorid from URL
import axios from "axios";

const PaymentButton = () => {
  const { setPaymentCompleted } = useContext(PaymentContext);
  //   const { doctorid } = useParams(); // Get doctorid from URL
  const [doctor, setDoctor] = useState({});
  const location = useLocation();
  const urlParams = location.pathname.split("/");
  const doctorid = urlParams[urlParams.length - 1];
  console.log("doctorid", doctorid);

  const getUserData = async () => {
    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/doctor/getDoctorById",
        {
          doctorId: doctorid,
        },

        {
          headers: {
            Authorization: "Bearer " + localStorage.getItem("token"),
          },
        }
      );
      if (res.data.success) {
        setDoctor(res.data.data);
      }
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    getUserData();
  }, []);

  console.log("the doctor", doctor.feesPerConsultation);
  const config = {
    publicKey: "Key 8007758fd80140b79bfe3a41f1a78b5a",
    productIdentity: String(doctorid), // Use doctorid as the product identity
    productName: "Doctor Appointment",
    productUrl: window.location.href,
    eventHandler: {
      onSuccess(payload) {
        // hit merchant api for verifying payment
        fetch("http://localhost:8000/verifyKhaltiPayment", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        })
          .then((response) => response.json())
          .then((data) => {
            console.log(data);
          })
          .catch((error) => {
            console.error("Error:", error);
          });
        console.log(payload);
        setPaymentCompleted(true); // Update payment status in context
      },
      onError(error) {
        console.log(error);
      },
      onClose() {
        console.log("Widget is closing");
      },
    },
  };

  const checkout = new KhaltiCheckout(config);

  return (
    <>
      <Card>
        <img
          src="/khalti.jpg"
          alt="Khalti Logo"
          style={{
            height: "100px",
            marginRight: "8px",
            width: "100px",
          }}
        />
        <Button
          type="primary"
          onClick={() =>
            checkout.show({ amount: doctor?.feesPerConsultation * 100 })
          }
          style={{
            display: "flex",
            alignItems: "center",
            padding: "10px 20px",
            backgroundColor: "#00bfae", // Khalti's brand color
            borderColor: "#00bfae", // Match button border color with background
          }}
        >
          <span>Pay with Khalti</span>
        </Button>
      </Card>
    </>
  );
};

export default PaymentButton;
