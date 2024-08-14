import React, { createContext, useState } from "react";

export const PaymentContext = createContext();

export const PaymentProvider = ({ children }) => {
  const [isPaymentCompleted, setPaymentCompleted] = useState(false);

  return (
    <PaymentContext.Provider
      value={{ isPaymentCompleted, setPaymentCompleted }}
    >
      {children}
    </PaymentContext.Provider>
  );
};
