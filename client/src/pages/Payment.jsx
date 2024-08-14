import React from "react";
import { Button, message, Steps, theme, Form, Input, Checkbox } from "antd";

const Payment = () => {
  return (
    <>
      <Form layout="vertical">
        <Form.Item label="Card Number">
          <Input placeholder="e.g., 1234 5678 9876 5432" />
        </Form.Item>
        <Form.Item label="Expiry Date">
          <Input placeholder="e.g., 08/24" />
        </Form.Item>
        <Form.Item label="CVV">
          <Input placeholder="e.g., 123" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" block>
            Make Payment
          </Button>
        </Form.Item>
      </Form>
    </>
  );
};

export default Payment;
