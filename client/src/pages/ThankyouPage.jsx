import React from "react";
import { Result, Button } from "antd";
import { useNavigate } from "react-router-dom";

const ThankYouPage = () => {
  //   const history = useHistory();
  const navigate = useNavigate();

  const handleHomeClick = () => {
    // navigate.push("/"); // Redirect to home or another page
    navigate("/");
  };

  return (
    <div style={{ padding: "50px", textAlign: "center" }}>
      <Result
        status="success"
        title="Booking Successful!"
        subTitle="Thank you for booking your appointment with us. We look forward to seeing you."
        extra={[
          <Button type="primary" key="home" onClick={handleHomeClick}>
            Back to Home
          </Button>,
        ]}
      />
    </div>
  );
};

export default ThankYouPage;
