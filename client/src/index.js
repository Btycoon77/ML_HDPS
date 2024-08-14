import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import "antd/dist/reset.css";
import { Provider } from "react-redux";
import store from "./redux/store";
import { FormProvider } from "../src/algorithm/FormContext";
import { PaymentProvider } from "../src/pages/PaymentContext";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <Provider store={store}>
    <FormProvider>
      <PaymentProvider>
        <React.StrictMode>
          <App />
        </React.StrictMode>
      </PaymentProvider>
    </FormProvider>
  </Provider>
);
