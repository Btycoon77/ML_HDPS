import React, { useState, useContext } from "react";

import { Radio } from "antd";
import Layout from "../components/Layout";
import { Col, Form, Input, Row } from "antd";
import { message } from "antd";
import { useNavigate } from "react-router-dom";
import { FormContext } from "./FormContext";

const HeartDiseasePrediction = () => {
  const navigate = useNavigate();
  const { setFormData } = useContext(FormContext);

  const handleSubmit = async (values) => {
    // List of required fields
    const requiredFields = [
      { field: selectedAge, name: "age" },
      { field: selectedSex, name: "sex" },
      { field: selectedYears, name: "years" },
      { field: selectedCHP, name: "chp" },
      { field: selectedHeight, name: "height" },
      { field: selectedWeight, name: "weight" },
      { field: selectedHeartrate, name: "heart rate" },
      { field: selectedBPsys, name: "systolic blood pressure" },
      { field: selectedBPdias, name: "diastolic blood pressure" },
      { field: selectedLDL, name: "LDL cholesterol" },
      { field: selectedIVSD, name: "IVSD" },
      { field: selectedSmoker, name: "smoking status" },
      { field: selectedFH, name: "family history" },
      { field: selectedActive, name: "physical activity" },
      { field: selectedLifeStyle, name: "lifestyle" },
      { field: selectedDiabetesStatus, name: "diabetes status" },
      { field: selectedHypertensionStatus, name: "hypertension status" },
      { field: selectedECG, name: "ECG pattern" },
      { field: selectedQwave, name: "Q-wave" },
      { field: selectedIHD, name: "ischemic heart disease" },
    ];

    // Validate if any required field is missing
    for (let i = 0; i < requiredFields.length; i++) {
      if (requiredFields[i].field == null || requiredFields[i].field === "") {
        // Display an error message and stop the function
        message.error(
          `Please fill in the ${requiredFields[i].name} field before submitting.`
        );
        return;
      }
    }
    //  proceed if the validation succeds

    const formData = {
      age: Number(selectedAge),
      sex: selectedSex,
      years: Number(selectedYears),
      chp: selectedCHP,
      height: Number(selectedHeight),
      weight: Number(selectedWeight),
      hr: Number(selectedHeartrate),
      bpsys: Number(selectedBPsys),
      bpdias: Number(selectedBPdias),
      ldl: Number(selectedLDL),
      ivsd:Number(selectedIVSD),
      smoke: selectedSmoker,
      fh: selectedFH,
      active: selectedActive,
      lifestyle: selectedLifeStyle,
      dm: selectedDiabetesStatus,
      htn: selectedHypertensionStatus,
      ecgpatt: selectedECG,
      qwave: selectedQwave,
      ihd: selectedIHD,
    };
    setFormData(formData);
    console.log("values of the form", formData);
    navigate("/predict-result");
  };

  //  validate zero or one;

  const [selectedAge, setSelectedAge] = useState(null);

  const handleAgeChange = (e) => {
    setSelectedAge(e.target.value);
  };

  const validateAge = (rule, value) => {
    if (value == null) {
      return Promise.reject(new Error("Please enter your age"));
    }
    if (value < 1 || value > 100) {
      return Promise.reject("Age must be between 1 and 100");
    }
    return Promise.resolve();
  };

  const [selectedSex, setSelectedSex] = useState(null);

  const handleSexChange = (e) => {
    setSelectedSex(e.target.value);
  };
  //  validate sex
  const validateSex = (rule, value) => {
    if (value == null) {
      return Promise.reject(new Error("Please enter your gender"));
    }
    if (value !== 0 && value !== 1) {
      return Promise.reject("Sex must be either 'Male' or 'Female'.");
    }
    return Promise.resolve();
  };

  //  validate smoker status

  const [selectedSmoker, setSelectedSmoker] = useState(null);

  const handleSmokerChange = (e) => {
    setSelectedSmoker(e.target.value);
  };

  const validateSmokerStatus = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("Smoker status must be either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate Smoking years.

  const [selectedYears, setSelectedYears] = useState(null);

  const handleYearChange = (e) => {
    setSelectedYears(e.target.value);
  };

  //  validate height.

  const [selectedHeight, setSelectedHeight] = useState(null);

  const handleHeightChange = (e) => {
    setSelectedHeight(e.target.value);
  };

  //  validate weight.

  const [selectedWeight, setSelectedWeight] = useState(null);

  const handleWeightChange = (e) => {
    setSelectedWeight(e.target.value);
  };

  // validate bdsys

  const [selectedBPsys, setSelectedBPsys] = useState(null);

  const handleBPsysChange = (e) => {
    setSelectedBPsys(e.target.value);
  };

  // validate bddias

  const [selectedBPdias, setSelectedBPdias] = useState(null);

  const handleBPdiasChange = (e) => {
    setSelectedBPdias(e.target.value);
  };

  //  validate ldl
  const [selectedLDL, setSelectedLDL] = useState(null);

  const handleLDLChange = (e) => {
    setSelectedLDL(e.target.value);
  };

  const validateLDL = (rule, value) => {
    if (value == null) {
      return Promise.reject(new Error("Please enter LDL "));
    }
    if (value < 0 || value > 500) {
      return Promise.reject("enter ldl cholestorerl level between 0 and 500");
    }
    return Promise.resolve();
  };

  //  validate chp

  const [selectedCHP, setSelectedCHP] = useState(null);

  const handleCHPChange = (e) => {
    setSelectedCHP(e.target.value);
  };

  const validateChp = (rule, value) => {
    // If the value is undefined or null, reject with an error message
    if (value == null) {
      return Promise.reject(new Error("Please select a chest pain type"));
    }

    // Use a switch statement to check for valid values
    switch (value) {
      case 1:
      case 2:
      case 3:
      case 4:
        return Promise.resolve(); // Valid value, resolve the promise
      default:
        return Promise.reject(new Error("Invalid chest pain type selected"));
    }
  };

  //  validate family history
  const [selectedFH, setSelectedFH] = useState(null);

  const handleFh = (e) => {
    setSelectedFH(e.target.value);
  };

  const validateFh = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter family history either 0 or 1");
    }
    return Promise.resolve();
  };
  //  validate activeness

  const [selectedActive, setSelectedActive] = useState(null);

  const handleActiveChange = (e) => {
    setSelectedActive(e.target.value);
  };

  const validateActive = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter activeness either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate lifestyle

  const [selectedLifeStyle, setSelectedLifeStyle] = useState(null);

  const handleLifeStyleChange = (e) => {
    setSelectedLifeStyle(e.target.value);
  };

  const validateLifeStyle = (rule, value) => {
    // If the value is undefined or null, reject with an error message
    if (value == null) {
      return Promise.reject(new Error("Please select your lifestyle"));
    }

    // Use a switch statement to check for valid values
    switch (value) {
      case 1:
      case 2:
      case 3:
      case 4:
        return Promise.resolve(); // Valid value, resolve the promise
      default:
        return Promise.reject(new Error("Invalid lifestyle type selected"));
    }
  };

  //  validate ihd

  const [selectedIHD, setSelectedIHD] = useState(null);

  const handleIHDChange = (e) => {
    setSelectedIHD(e.target.value);
  };

  const validateIhd = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter presence of ihd as either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate Heartrate

  const [selectedHeartrate, setSelectedHeartrate] = useState(null);

  const handleHeartrate = (e) => {
    setSelectedHeartrate(e.target.value);
  };

  const validateHeartRate = (rule, value) => {
    if (value < 50 || value > 120) {
      return Promise.reject("enter activeness between 50 and 120");
    }
    return Promise.resolve();
  };

  //  validate diabetes status

  const [selectedDiabetesStatus, setSelectedDiabetesStatus] = useState(null);

  const handleDiabetesStatusChange = (e) => {
    setSelectedDiabetesStatus(e.target.value);
  };

  const validateDiabetesStatus = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter diabetes either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate htn

  const [selectedHypertensionStatus, setSelectedHypertensionStatus] =
    useState(null);

  const handleHypertensionStatusChange = (e) => {
    setSelectedHypertensionStatus(e.target.value);
  };

  const validateHTN = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter hypertension status as  either 0 or 1");
    }
    return Promise.resolve();
  };

  //  validate IVSD

  const [selectedIVSD, setSelectedIVSD] = useState(null);

  const handleIVSD = (e) => {
    setSelectedIVSD(e.target.value);
  };

  const validateIVSD = (rule, value) => {
    if (value < 0 || value > 2) {
      return Promise.reject(
        "enter intraventricular septal thickenss  between 0 or 2"
      );
    }
    return Promise.resolve();
  };

  //  validate ecpPatt

  const [selectedECG, setSelectedECG] = useState(null);

  const handleECGChange = (e) => {
    setSelectedECG(e.target.value);
  };

  const validateECGpatt = (rule, value) => {
    // If the value is undefined or null, reject with an error message
    if (value == null) {
      return Promise.reject(new Error("Please select your ECG-patt type"));
    }

    // Use a switch statement to check for valid values
    switch (value) {
      case 1:
      case 2:
      case 3:
      case 4:
        return Promise.resolve(); // Valid value, resolve the promise
      default:
        return Promise.reject(
          new Error("Invalid chest ECG-patt type selected")
        );
    }
  };

  //  validate qwave

  const [selectedQwave, setSelectedQwave] = useState(null);

  const handleQwaveChange = (e) => {
    setSelectedQwave(e.target.value);
  };

  const validateQwave = (rule, value) => {
    if (value < 0 || value > 1) {
      return Promise.reject("enter Qwave as either 0 or 1");
    }
    return Promise.resolve();
  };

  return (
    <>
      <Layout>
        <h1 className="text-center">Heart Disease Prediction</h1>
        <Form layout="vertical" onFinish={handleSubmit} className="m-3">
          <Row gutter={20}>
            <Col xs={24} md={24} lg={8}>
              {/* <h4>Personal Health details:</h4> */}
              <Form.Item
                label="Age"
                name="age"
                required
                rules={[
                  {
                    required: true,
                    min: 1,
                    max: 100,
                    // message: "Age must be greater than 0 and less than 100",
                    validator: validateAge,
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder="enter your age"
                  onChange={handleAgeChange}
                  value={selectedAge}
                />
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="sex"
                name="sex"
                required
                rules={[
                  { required: true, message: "please enter your sex" },
                  {
                    validator: validateSex,
                  },
                ]}
              >
                <Radio.Group onChange={handleSexChange} value={selectedSex}>
                  <Radio value={1}>Male</Radio>
                  <Radio value={0}>Female</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Smoker"
                name="smoke"
                required
                rules={[
                  { required: true, message: "Please select smoker status" },
                  {
                    validator: validateSmokerStatus,
                  },
                ]}
              >
                <Radio.Group
                  onChange={handleSmokerChange}
                  value={selectedSmoker}
                >
                  <Radio value={1}>Yes</Radio>
                  <Radio value={0}>No</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Years"
                name="years"
                required
                rules={[{ required: true }]}
              >
                <Input
                  type="number"
                  min={0}
                  max={100}
                  placeholder="Enter years of smoking (0 if non-smoker"
                  onChange={handleYearChange}
                  value={selectedYears}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Ldl-Cholesterol level"
                name="ldl"
                required
                rules={[{ required: true }, { validator: validateLDL }]}
              >
                <Input
                  type="number"
                  max={200}
                  placeholder="Enter LDL cholesterol level: "
                  onChange={handleLDLChange}
                  value={selectedLDL}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Chest-pain Type"
                name="chp"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateChp,
                  },
                ]}
              >
                <Radio.Group onChange={handleCHPChange} value={selectedCHP}>
                  <Radio value={1}>Typical Angina</Radio>
                  <Radio value={2}>Atypical Angina</Radio>
                  <Radio value={3}>Non-anginal pain</Radio>
                  <Radio value={4}>Asymptomatic</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="height"
                name="height"
                required
                rules={[
                  { required: true, message: "please enter your height" },
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter height in cm"
                  min={60}
                  max={245}
                  onChange={handleHeightChange}
                  value={selectedHeight}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Weight"
                name="weight"
                required
                rules={[
                  { required: true, message: "please enter your weight" },
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter weight in kg:"
                  min={2}
                  max={500}
                  onChange={handleWeightChange}
                  value={selectedWeight}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Family History"
                name="fh"
                required
                rules={[
                  {
                    required: true,
                  },
                  {
                    validator: validateFh,
                  },
                ]}
              >
                <Radio.Group onChange={handleFh} value={selectedFH}>
                  <Radio value={1}>Yes</Radio>
                  <Radio value={0}>No</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Active"
                name="active"
                required
                rules={[
                  { required: true, message: "please enter your activeness" },
                  {
                    validator: validateActive,
                  },
                ]}
              >
                <Radio.Group
                  onChange={handleActiveChange}
                  value={selectedActive}
                >
                  <Radio value={1}>Yes</Radio>
                  <Radio value={0}>No</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>

            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Lifestyle"
                name="lifestyle"
                required
                rules={[
                  {
                    required: true,
                  },
                  {
                    validator: validateLifeStyle,
                  },
                ]}
              >
                <Radio.Group
                  onChange={handleLifeStyleChange}
                  value={selectedLifeStyle}
                >
                  <Radio value={1}>City</Radio>
                  <Radio value={2}>Town</Radio>
                  <Radio value={3}>Village</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Ischemic Heart Disease"
                name="ihd"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateIhd,
                  },
                ]}
              >
                <Radio.Group onChange={handleIHDChange} value={selectedIHD}>
                  <Radio value={1}>Yes History</Radio>
                  <Radio value={0}>No History</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Heart Rate"
                name="hr"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateHeartRate,
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter resting heart rate"
                  min={50}
                  max={220}
                  onChange={handleHeartrate}
                  value={selectedHeartrate}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Diabetes status"
                name="dm"
                required
                rules={[
                  {
                    required: true,
                  },
                  {
                    validator: validateDiabetesStatus,
                  },
                ]}
              >
                <Radio.Group
                  onChange={handleDiabetesStatusChange}
                  value={selectedDiabetesStatus}
                >
                  <Radio value={1}>Diabetic</Radio>
                  <Radio value={0}>Non-Diabetic</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Bpsys"
                name="bpsys"
                required
                rules={[{ required: true }]}
              >
                <Input
                  type="number"
                  placeholder="Enter systolic blood pressure "
                  onChange={handleBPsysChange}
                  value={selectedBPsys}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Bpdias"
                name="bpdias"
                required
                rules={[{ required: true }]}
              >
                <Input
                  type="number"
                  placeholder="Enter diastolic blood pressure "
                  onChange={handleBPdiasChange}
                  value={selectedBPdias}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Hypertension status"
                name="htn"
                required
                rules={[
                  {
                    required: true,
                  },
                  {
                    validator: validateHTN,
                  },
                ]}
              >
                <Radio.Group
                  onChange={handleHypertensionStatusChange}
                  value={selectedHypertensionStatus}
                >
                  <Radio value={1}>Normal</Radio>
                  <Radio value={0}>Abnormal</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="IVSD"
                name="ivsd"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateIVSD,
                  },
                ]}
              >
                <Input
                  type="number"
                  placeholder="Enter interventricular septal thickness:"
                  min={0}
                  max={2}
                  onChange={handleIVSD}
                  value={selectedIVSD}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="ECGpatt"
                name="ecgpatt"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateECGpatt,
                  },
                ]}
              >
                <Radio.Group onChange={handleECGChange} value={selectedECG}>
                  <Radio value={1}>ST-Elevation</Radio>
                  <Radio value={2}>ST-Depression</Radio>
                  <Radio value={3}>T-Inversion</Radio>
                  <Radio value={4}>Normal</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={24} lg={8}>
              <Form.Item
                label="Qwave"
                name="qwave"
                required
                rules={[
                  { required: true },
                  {
                    validator: validateQwave,
                  },
                ]}
              >
                <Radio.Group onChange={handleQwaveChange} value={selectedQwave}>
                  <Radio value={1}>Present</Radio>
                  <Radio value={0}>Absent</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
          </Row>
          <div className="d-flex justify-content-end">
            <button onClick={handleSubmit} className="btn btn-primary form-btn">
              Submit
            </button>
          </div>
        </Form>
      </Layout>
    </>
  );
};

export default HeartDiseasePrediction;
