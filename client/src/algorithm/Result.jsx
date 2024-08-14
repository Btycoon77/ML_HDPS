import React, { useContext, useEffect, useState } from "react";
import Layout from "../components/Layout";
import { Card, Col, Row } from "antd";
import Title from "antd/es/typography/Title";
import axios from "axios";
import "./result.css";
import { useSelector } from "react-redux";
import { FormContext } from "./FormContext";

const Result = () => {
  const [predictions, setPrediction] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const { user } = useSelector((state) => state.user);
  // console.log("State of the form",state);
  const { formData } = useContext(FormContext);
  console.log("form data", formData);

  const handleServices = async (values) => {
    try {
      console.log("clicked on prediction");
      console.log(values);
      const response = await axios.post(
        "http://localhost:8000/predict",
        values,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );
      const prediction = response.data.predictions;
      const accuracies = response.data.accuracies;

      console.log("the response data is ", response.data);
      // console.log("the response data is ", response.data.data);
      setPrediction(prediction);
      console.log("predictions of algo are", prediction);
      // let accuracy = response.data.test_accuracy_scratch;
      // let accuracyPercentage = accuracy * 100;
      // setAccuracy(accuracyPercentage);
      // console.log(accuracyPercentage);
      setAccuracy(accuracies);
      console.log("accuracies of algo are", accuracies);
    } catch (error) {
      console.error(error);
    }
  };

  // handling the accuracy result

  // Call handleServices when the component mounts
  // useEffect(() => {
  //   if (formData) { // Check if formData is available
  //     handleServices(formData); // Pass formData to handleServices
  //   }
  // }, [formData]); // Add formData as a dependency

  useEffect(() => {
    const fetchData = async () => {
      if (formData) {
        setLoading(true);
        try {
          // Perform both API calls in parallel
          await Promise.all([
            handleServices(formData),
            // handleEvaluation(formData),
          ]);
        } catch (err) {
          setError(err);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchData();
  }, [formData]);

  console.log("the final accuracy is", accuracy);

  return (
    <Layout className="result-layout">
      <Row justify="center">
        <Col xs={24} md={18} lg={12}>
          <Card bordered={false} className="result-card">
            <Title level={2} className="result-title">
              Heart Disease Prediction Result
            </Title>
            <div className="result-info">
              <p>
                <b>Username:</b> {user?.name}
              </p>

              <p>
                <b>Age:</b> {formData?.age}
              </p>
              <p>
                <b>Sex:</b> {formData?.sex === 1 ? "Male" : "Female"}
              </p>
              <p>
                <b>
                  <u>Prediction Result</u>
                </b>
              </p>
              {predictions && (
                <div className="result-predictions">
                  <p>
                    <b>Random Forest Prediction:</b>
                    {predictions.random_forest === 1
                      ? " Sorry! You have Heart Disease"
                      : "No Heart Disease"}
                  </p>
                  <p>
                    <b>SVM Prediction:</b>
                    {predictions.svm === 1
                      ? " Sorry! You have Heart Disease"
                      : "No Heart Disease"}
                  </p>
                  <p>
                    <b>KNN Prediction:</b>
                    {predictions.knn === 1
                      ? " Sorry! You have Heart Disease"
                      : "No Heart Disease"}
                  </p>
                </div>
              )}
            </div>
            {accuracy && (
              <div className="result-accuracies">
                <p>
                  <b>Random Forest Accuracy:</b> {accuracy.random_forest}%
                </p>
                <p>
                  <b>SVM Accuracy:</b> {accuracy.svm}%
                </p>
                <p>
                  <b>KNN Accuracy:</b> {accuracy.knn}%
                </p>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </Layout>
  );
};

export default Result;
