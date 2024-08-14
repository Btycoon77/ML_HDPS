import React, { useContext, useState } from "react";
import { PaymentContext } from "./PaymentContext";
import CheckAvailability from "./CheckAvailability";
import Payment from "./Payment";
import BookingPage from "./BookingPage";
import { Steps, Button, message } from "antd";
import Layout from "../components/Layout";
import PaymentButton from "../components/PaymentButton";
import ThankYouPage from "./ThankyouPage";

const steps = [
  { title: "Check Availability", content: <CheckAvailability /> },
  { title: "Book Appointment", content: <BookingPage /> },
  // { title: "Payment", content: <PaymentButton /> },
  { title: "Thankyou Page", content: <ThankYouPage /> },
];

const BookingProcess = () => {
  const { isPaymentCompleted } = useContext(PaymentContext);
  const [current, setCurrent] = useState(0);

  const next = () => {
    // if (current === 1 && !isPaymentCompleted) {
    //   message.error(
    //     "You must complete the payment before booking an appointment."
    //   );
    //   return;
    // }
    setCurrent(current + 1);
  };

  const prev = () => setCurrent(current - 1);

  return (
    <>
      <Layout>
        <Steps
          current={current}
          items={steps.map((item) => ({ key: item.title, title: item.title }))}
        />
        <div style={{ marginTop: 24 }}>{steps[current].content}</div>
        <div style={{ marginTop: 24 }}>
          {current < steps.length - 1 && (
            <Button type="primary" onClick={next}>
              Next
            </Button>
          )}
          {current === steps.length - 1 && (
            <Button
              type="primary"
              onClick={() =>
                message.success("Appointment booked successfully!")
              }
            >
              Done
            </Button>
          )}
          {current > 0 && (
            <Button style={{ margin: "0 8px" }} onClick={prev}>
              Previous
            </Button>
          )}
        </div>
      </Layout>
    </>
  );
};

export default BookingProcess;
