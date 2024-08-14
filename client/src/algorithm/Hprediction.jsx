import React, { useState } from "react";
import axios from "axios";
import { message } from "antd";
import Layout from "../components/Layout";
import { Col, Form, Input, Row, Card,Radio } from "antd";

const Hprediction = () => {
  const [prediction, setPrediction] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const handleSubmit = async (values) => {
    try {
      console.log("clicked on prediction");
      const response = await axios.post(
        "http://localhost:8000/predict",
        values,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      const prediction = response.data.prediction_scratch;
      setPrediction(prediction);
      console.log(prediction);
      let accuracy = response.data.test_accuracy_scratch;
      let accuracyPercentage = accuracy * 100;
      setAccuracy(accuracyPercentage);
      console.log(accuracyPercentage);
    } catch (error) {
      console.error(error);
    }
  };

  //  validate zero or one;

  const validateAge = (rule, value) => {
    if (value < 1 || value > 100) {
      return Promise.reject("Age must be between 1 and 100");
    }
    return Promise.resolve();
  };

  //  validate sex
  const validateSex = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("Sex must be either 0 or 1");
    }
    return Promise.resolve();
  };
  //  validate contractile pressure
  const validateCp = (rule, value) => {
    if (value < 0 || value > 4) {
      return Promise.reject("Please select in a range of 0 to 4");
    }
    return Promise.resolve();
  };

  //  validate trestbps

  const validatetrestbps = (rule, value) => {
    if (value <94 || value >201) {
      return Promise.reject("enter  cholestorerl level between 94 and 201");
    }
    return Promise.resolve();
  };

  //  validate cholestrol level

  const validateCholestrol = (rule, value) => {
    if (value < 126 || value > 565) {
      return Promise.reject("cholestrol level must be betweein 126 and 565");
    }
    return Promise.resolve();
  };

  //  validate fasting glucose level

  const validateFbs = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter fbs  either 0 or 1");
    }
    return Promise.resolve();
  };
  //  validate restecg

  const validateRestecg = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter restecg either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate thalach

  const validateThalach = (rule, value) => {
    if (value < 70 || value > 202) {
      return Promise.reject("enter thalach between 71 and 202");
    }
    return Promise.resolve();
  };

  //  validate exang

  const validateExang = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter exang test as either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate old peak

  const validateOldPeak = (rule, value) => {
    if (value <0  || value >= 6.2) {
      return Promise.reject("enter activeness between 0 and 6.2");
    }
    return Promise.resolve();
  };

  //  validate slope

  const validateSlope = (rule, value) => {
    if (value < 0 || value >1) {
      return Promise.reject("enter slope in the range  of 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate calcium concentration

  const validateCalcium = (rule, value) => {
    if (value < 90 || value >200) {
      return Promise.reject("enter proper calcium concentration value");
    }
    return Promise.resolve();
  };

  //  validate thal

  const validateThal = (rule, value) => {
    if (value < 1 || value > 3) {
      return Promise.reject(
        "enter thalassemia in the range of 1,2,3"
      );
    }
    return Promise.resolve();
  };


  return (
    <>
      <Layout>
        <h1 className="text-center">Heart Disease Prediction</h1>
        <Form layout="vertical" onFinish={handleSubmit} className="m-3">
          <Row gutter={20}>
            {/* Age */}
            <Col xs={24} md={24} lg={8}>
              <h4>Personal Health details:</h4>
              <Form.Item
                label="Age"
                name="age"
                required
                rules={[
                    { required: true, message: "Please enter your age" },
                    {
                        validator: validateAge,
                    },
                    ]}
              >
                <Input type="number" placeholder="enter your age" />
              </Form.Item>
            </Col>
            {/* Sex */}
            <Col xs={24} md={24} lg={8}>
            <Form.Item
                label="Sex"
                name="sex"
                required
                rules={[
                { required: true, message: "Please select your gender" },
                {
                    validator: validateSex,
                },
                ]}
            >
                <Radio.Group>
                <Radio value={1}>Male</Radio>
                <Radio value={0}>Female</Radio>
                </Radio.Group>
            </Form.Item>
            </Col>
            {/* contractile pressure */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Contractile Pressure"
                name="cp"
                required
                rules={[
                  { required: true, message: "Please enter your cp" },
                  {
                    validator: validateCp,
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder=" Enter your contractile pressure"
                  min={0}
                  max={3}
                  step={1}
                />
              </Form.Item>
            </Col>
            {/* trestbps */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Trestbps"
                name="trestbps"
                required
                rules={[{ required: true },{validator:validatetrestbps}]}
              >
                <Input
                  type="number"
                  min={94}
                  max={200}
                  placeholder="Enter your resting blood pressure"
                />
              </Form.Item>
            </Col>
            {/* cholestrol */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Cholestrol"
                name="ch"
                required
                rules={[{ required: true },{validator:validateCholestrol}]}
              >
                <Input
                  type="number"
                  min={110}
                  max={564}
                  placeholder="Enter  cholesterol level"
                />
              </Form.Item>
            </Col>
            {/* Fating Glucose Level */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="fbs"
                name="fbs"
                required
                rules={[
                  { required: true },
                    {validator:validateFbs}
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter fasting glucose level"
                  min={0}
                  max={1}
                  step={1}
                />
              </Form.Item>
            </Col>
                {/* RestEcg */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="restecg"
                name="restecg"
                required
                rules={[
                  { required: true, message: "please enter your resting ecg test" },{validator:validateRestecg},
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your resting ecg test"
                  min={0}
                  max={1}
                  step={1}
                />
              </Form.Item>
            </Col>
            {/* Thalach */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Thalach"
                name="thalach"
                required
                rules={[
                  { required: true, message: "please enter your maximum heart rate achieved" },{validator:validateThalach}
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your maximum heart rate achieved"
                  min={71}
                  max={202}
                />
              </Form.Item>
            </Col>
            {/* Exang */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Exang test"
                name="exang"
                required
                rules={[
                  {
                    required: true,
                  },
                  {validator:validateExang}
                 
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your exang test 0 or 1"
                  min={0}
                  max={1}
                  step={1}
                />
              </Form.Item>
            </Col>
            {/* old peak */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Old peak"
                name="oldpeak"
                required
                rules={[
                  { required: true, message: "please enter your max systolic or distolic pressure" },
                  {validator:validateOldPeak}
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your max systolic or distolic pressure"
                  min={0}
                  max={6.2}
                  
                />
              </Form.Item>
            </Col>

                {/* slope */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Slope"
                name="slope"
                required
                rules={[
                  {
                    required: true,
                  },
                 {
                    validator:validateSlope
                 }
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your change in heart rate over time"
                  min={0}
                  max={2}
                  step={1}
                />
              </Form.Item>
            </Col>
            {/* calcium */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Calcium concentration"
                name="ca"
                required
                rules={[
                  { required: true },
                 {validator:validateCalcium}
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter concentration range of calcium in blood"
                  min={90}
                  max={200}
                  
                />
              </Form.Item>
            </Col>

            {/* Thal */}
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Blood Disorder Thalassemia"
                name="thal"
                required
                rules={[
                  { required: true },
                  {validator:validateThal}
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter your status of blood disorder"
                  min={1}
                  max={3}
                  step={1}
                />
              </Form.Item>
            </Col>

            
          </Row>
          <div className="d-flex justify-content-end">
            <button onClick={handleSubmit} className="btn btn-primary form-btn">
              Submit
            </button>
          </div>
        </Form>
        {prediction !== undefined && (
          <div>
            {prediction === 0 ? (
              <>
                {message.success(
                  `User does not have heart disease and percentage of accuracy is ${accuracy}`
                )}
                {
                  <Card
                    title="Heart Disease prediction Result"
                    bordered={true}
                    style={{ width: 300 }}
                  >
                    <p>
                      Based on the input, it is predicted that the user does not
                      have heart disease.
                    </p>
                    <p>{`The prediction accuracy is ${accuracy}`}</p>
                  </Card>
                }
              </>
            ) : (
              <>
                {message.warning(
                  `User has heart disease and percentage of accuracy is ${accuracy}`
                )}
                <Card
                  title="Heart Disease prediction Result"
                  bordered={true}
                  style={{ width: 300 }}
                >
                  <p>It is predicted that the user has heart disease.</p>
                  <p>{`The prediction accuracy is ${accuracy}`}</p>
                </Card>
              </>
            )}
          </div>
        )}
      </Layout>
    </>
  );
};

export default Hprediction;
