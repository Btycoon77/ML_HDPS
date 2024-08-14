// // const tf = require("@tensorflow/tfjs");
// // const fs = require("fs");
// // const csv = require("csv-parser");
// // const path = require("path");

// // const csvFilePath = path.join(__dirname, "..", "ECG-Dataset.csv");

// // // Load the CSV file
// // const loadData = async () => {
// //   const data = [];
// //   return new Promise((resolve, reject) => {
// //     fs.createReadStream(csvFilePath)
// //       .pipe(csv({ headers: true }))
// //       .on("data", (row) => {
// //         const convertedRow = {};
// //         for (const [key, value] of Object.entries(row)) {
// //           convertedRow[key] = parseFloat(value);
// //         }
// //         data.push(convertedRow);
// //       })
// //       .on("end", () => {
// //         const features = data.map((row) => Object.values(row));
// //         const labels = data.map((row) => (row.target === "1" ? 1 : 0));
// //         resolve({ features, labels });
// //       })
// //       .on("error", (error) => {
// //         reject(error);
// //       });
// //   });
// // };

// // // Define and train the model
// // const trainModel = async (XTrainScaled, yTrain) => {
// //   const model = tf.sequential();
// //   model.add(
// //     tf.layers.dense({
// //       units: 16,
// //       activation: "relu",
// //       inputShape: [XTrainScaled[0].length],
// //     })
// //   );
// //   model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
// //   model.compile({
// //     optimizer: tf.train.adam(),
// //     loss: "binaryCrossentropy",
// //     metrics: ["accuracy"],
// //   });

// //   const history = await model.fit(tf.tensor2d(XTrainScaled), tf.tensor1d(yTrain), {
// //     epochs: 10,
// //     batchSize: 32,
// //   });

// //   return model;
// // };

// // // Save the model
// // const saveModel = async (model) => {
// //   const modelDir = path.join(__dirname, "models");
// //   const modelSavePath = path.join(modelDir, "model.json");

// //   // Create the model directory if it doesn't exist
// //   if (!fs.existsSync(modelDir)) {
// //     fs.mkdirSync(modelDir);
// //   }

// //   // Save the model as JSON
// //   const modelJson = model.toJSON();
// //   fs.writeFileSync(modelSavePath, JSON.stringify(modelJson));

// //   console.log("Model saved successfully.");
// // };

// // // Main controller function
// // const HeartDiseasePredictionController = async (req, res) => {
// //   try {
// //     // Load the data
// //     const { features, labels } = await loadData();

// //     // Split the data into training and testing sets
// //     const splitIndex = Math.floor(features.length * 0.8);
// //     const XTrain = features.slice(0, splitIndex);
// //     const yTrain = labels.slice(0, splitIndex);
// //     const XTest = features.slice(splitIndex);
// //     const yTest = labels.slice(splitIndex);

// //     // Preprocess the data (scaling, etc.)
// //     // ... (add your preprocessing code here) ...

// //     // Train the model
// //     const model = await trainModel(XTrain, yTrain);

// //     // Save the model
// //     await saveModel(model);

// //     // Return success response
// //     res.json({ message: "Model trained and saved successfully." });
// //   } catch (error) {
// //     console.error("Error training and saving the model");

// //   }
// // }

// // module.exports = HeartDiseasePredictionController;

// const tf = require("@tensorflow/tfjs");
// const fs = require("fs");
// const csv = require("csv-parser");
// const path = require("path");

// const csvFilePath = path.join(__dirname, "..", "ECG-Dataset.csv");

// // Load the CSV file and preprocess the data
// const loadData = async () => {
//   const data = [];
//   return new Promise((resolve, reject) => {
//     fs.createReadStream(csvFilePath)
//       .pipe(csv({ headers: true }))
//       .on("data", (row) => {
//         const convertedRow = {};
//         for (const [key, value] of Object.entries(row)) {
//           convertedRow[key] = parseFloat(value);
//         }
//         data.push(convertedRow);
//       })
//       .on("end", () => {
//         const features = data.map((row) => {
//           return [
//             row.age,
//             row.sex === "male" ? 1 : 0,
//             row.smoke === "yes" ? 1 : 0,
//             row.years,
//             row.ldl,
//             row.chp,
//             row.height,
//             row.weight,
//             row.fh === "yes" ? 1 : 0,
//             row.active,
//             row.lifestyle,
//             row.ihd === "yes" ? 1 : 0,
//             row.hr,
//             row.dm === "yes" ? 1 : 0,
//             row.bpsys,
//             row.bpdias,
//             row.htn === "yes" ? 1 : 0,
//             row.ivsd,
//             row.ecgpatt,
//             row.qwave
//           ];
//         });
//         const labels = data.map((row) => (row.target === "1" ? 1 : 0));
//         resolve({ features, labels });
//       })
//       .on("error", (error) => {
//         reject(error);
//       });
//   });
// };

// // Define and train the model
// const trainModel = async (XTrainScaled, yTrain) => {
//   const model = tf.sequential();
//   model.add(
//     tf.layers.dense({
//       units: 16,
//       activation: "relu",
//       inputShape: [XTrainScaled[0].length],
//     })
//   );
//   model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
//   model.compile({
//     optimizer: tf.train.adam(),
//     loss: "binaryCrossentropy",
//     metrics: ["accuracy"],
//   });

//   const history = await model.fit(tf.tensor2d(XTrainScaled), tf.tensor1d(yTrain), {
//     epochs: 10,
//     batchSize: 32,
//   });

//   return model;
// };

// // Save the model
// const saveModel = async (model) => {
//   const modelDir = path.join(__dirname,"..", "models");
//   const modelSavePath = path.join(modelDir, "model.json");

//   // Create the model directory if it doesn't exist
//   if (!fs.existsSync(modelDir)) {
//     fs.mkdirSync(modelDir);
//   }

//   // Save the model as JSON
//   const modelJson = model.toJSON();
//   fs.writeFileSync(modelSavePath, JSON.stringify(modelJson));

//   console.log("Model saved successfully.");
// };

// // Load the trained model
// const loadModel = async () => {
//   const modelDir = path.join(__dirname, "models");
//   const modelPath = path.join(modelDir, "model.json");

//   if (!fs.existsSync(modelPath)) {
//     throw new Error("Model not found. Please train the model first.");
//   }

//   const model = await tf.loadLayersModel("http://localhost:8000/api/v1/user/model");
//   return model;
// };

// // Preprocess user input
// const preprocessInput = (input) => {
//   return {
//     age: input.age,
//     sex: input.sex === "male" ? 1 : 0,
//     smoke: input.smoke === "yes" ? 1 : 0,
//     years: input.years,
//     ldl: input.ldl,
//     chp: input.chp,
//     height: input.height,
//     weight: input.weight,
//     fh: input.fh === "yes" ? 1 : 0,
//     active: input.active,
//     lifestyle: input.lifestyle,
//     ihd: input.ihd === "yes" ? 1 : 0,
//     hr: input.hr,
//     dm: input.dm === "yes" ? 1 : 0,
//     bpsys: input.bpsys,
//     bpdias: input.bpdias,
//     htn: input.htn === "yes" ? 1 : 0,
//     ivsd: input.ivsd,
//     ecgpatt: input.ecgpatt,
//     qwave: input.qwave,
//     target: input.target,
//   };
// };

// // Perform prediction using the preprocessed user input
// const predictHeartDisease = async (preprocessedInput) => {
//   // Load the trained model
//   const model = await loadModel();

//   // Perform prediction using the model
//   const prediction = model.predict(tf.tensor2d([Object.values(preprocessedInput)])).dataSync()[0];

//   return prediction;
// };

// // Controller function to handle the request
// const predictHeartDiseaseController = async (req, res) => {
//   try {
//     // Get user input from req.body
//     const userInput = req.body;

//     // Preprocess user input
//     const preprocessedUserInput = preprocessInput(userInput);

//     // Perform prediction using the preprocessed user input
//     const prediction = await predictHeartDisease(preprocessedUserInput);

//     // Map the prediction to 1 or 0
//     const response = prediction >= 0.5 ? 1 : 0;

//     // Send the prediction as the response 
//     res.json({ prediction: response });
//   } catch (error) {
//     console.error("Error predicting heart disease:", error);
//     res.status(500).json({ error: "Error predicting heart disease" });
//   }
// };

// // Train and save the model
// const trainAndSaveModel = async () => {
//   try {
//     // Load the data
//     const { features, labels } = await loadData();

//     // Train the model
//     const model = await trainModel(features, labels);

//     // Save the model
//     await saveModel(model);
//   } catch (error) {
//     console.error("Error training and saving the model:", error);
//   }
// };

// // Call the trainAndSaveModel function to train and save the model
// trainAndSaveModel();

// module.exports = predictHeartDiseaseController;
