const express = require("express");
const morgan = require("morgan");
const connectDb = require("./config/connectDb");
const userRouter = require("./routes/userRoutes");
const dotenv = require("dotenv").config();
const cors = require("cors");
const adminRouter = require("./routes/adminRoute");
const doctorRouter = require("./routes/doctorRoute");
const path = require("path");
const { exec, spawn } = require("child_process");
const axios = require("axios");

const app = express();
const port = process.env.PORT || 8000;
const modelDir = path.join(__dirname, "models");

app.use(express.static(modelDir));
app.use(express.json());
app.use(morgan("dev"));
app.use(cors());
//  connect database;
connectDb();

// routes

app.use("/api/v1/user", userRouter);
app.use("/api/v1/admin", adminRouter);
app.use("/api/v1/doctor", doctorRouter);

// payment gateway

app.post("/verifyKhaltiPayment", async (req, res) => {
  const { token, amount } = req.body;

  try {
    const response = await axios.post(
      "https://khalti.com/api/v2/epayment/lookup/",
      {
        token,
        amount,
      },
      {
        headers: {
          Authorization: `Key ${process.env.YOUR_KHALTI_SECRET_KEY}`,
        },
      }
    );

    if (response.data.state.name === "Completed") {
      res.json({ success: true, message: "Payment verified successfully" });
    } else {
      res
        .status(400)
        .json({ success: false, message: "Payment verification failed" });
    }
  } catch (error) {
    res.status(500).json({ success: false, message: "Server Error" });
  }
});

//  for machine learing  prediction route

app.post("/predict", async (req, res) => {
  try {
    // Make a POST request to the Flask application running on a specific port

    //    create a  deep copy of the request bbody without circular reference;
    //  port:9000 (previously)
    const requestData = JSON.parse(JSON.stringify(req.body));
    const flaskResponse = await axios.post(
      "http://127.0.0.1:5000/predict",
      requestData
    );

    // Send the prediction response back to the client
    res.json(flaskResponse.data);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "An error occurred" });
  }
});

//  for getting the accuracy result

app.post("/evaluate", async (req, res) => {
  try {
    const requestData = JSON.parse(JSON.stringify(req.body));
    const flaskResponse = await axios.post(
      "http://127.0.0.1:5000/evaluate",
      requestData
    );

    // Send the prediction response back to the client
    res.json(flaskResponse.data);
  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "An error occurred" });
  }
});

// Spawn the Flask application as a child process
//  the prediction is actually being done from the myenv folder /app.py
// const flaskApp = spawn("python", ["myenv/app.py"]);
const flaskApp = spawn("python", ["myenv/api.py"]);

flaskApp.stdout.on("data", (data) => {
  console.log(`Flask App Output: ${data}`);
});

flaskApp.stderr.on("data", (data) => {
  console.error(`Flask App Error: ${data}`);
});

flaskApp.on("close", (code) => {
  console.log(`Flask App process exited with code ${code}`);
});

app.listen(port, () => {
  console.log(
    `server running at port ${port} in ${process.env.NODE_MODE} mode`
  );
});
