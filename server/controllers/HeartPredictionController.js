// const tf = require("@tensorflow/tfjs");
// const fs = require("fs");
// const csv = require("csv-parser");
// const path = require("path");

// const csvFilePath = path.join(__dirname, "..", "ECG-Dataset.csv");

// // Load the CSV file
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
//         const features = data.map((row) => Object.values(row));
//         const labels = data.map((row) => (row.target === "1" ? 1 : 0));
//         resolve({ features, labels });
//       })
//       .on("error", (error) => {
//         reject(error);
//       });
//   });
// };

// // Preprocess the data
// const createScaler = (XTrain) => {
//   const scaler = {
//     means: [],
//     stds: [],
//   };

//   for (let i = 0; i < XTrain[0].length; i++) {
//     const column = XTrain.map((row) => row[i]);
//     const mean = tf.tensor1d(column).mean().arraySync()[0];
//     const std = tf.tensor1d(column).sub(mean).pow(2).mean().sqrt().arraySync()[0];
//     scaler.means.push(mean);
//     scaler.stds.push(std);
//   }

//   return scaler;
// };

// const scaleData = (data, scaler) => {
//   const scaledData = data.map((row) => {
//     return row.map((value, i) => (value - scaler.means[i]) / scaler.stds[i]);
//   });

//   return scaledData;
// };

// // Train the model
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

//   const history = await model.fit(
//     tf.tensor2d(XTrainScaled),
//     tf.tensor1d(yTrain),
//     {
//       epochs: 10,
//       batchSize: 32,
//     }
//   );

//   return model;
// };

// // Save the model
// const saveModel = async (model) => {
//   const modelDir = path.join(__dirname,".." ,"models");
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

// // Main controller function
// const HeartDiseasePredictionController = async (req, res) => {
//   try {
//     // Load the data
//     const { features, labels } = await loadData();

//     // Split the data into training and testing sets
//     const splitIndex = Math.floor(features.length * 0.8);
//     const XTrain = features.slice(0, splitIndex);
//     const yTrain = labels.slice(0, splitIndex);
//     const XTest = features.slice(splitIndex);
//     const yTest = labels.slice(splitIndex);

//     // Preprocess the data (scaling, etc.)
//     const scaler = createScaler(XTrain);

//     const XTrainScaled = scaleData(XTrain, scaler);
//     const XTestScaled = scaleData(XTest, scaler);

//     // Train the model
//     const model = await trainModel(XTrainScaled, yTrain);

//     // Save the model
//     await saveModel(model);

//     // Preprocess user input
//     const userInput = {
//       age: parseFloat(req.body.age),
//       sex: req.body.sex === "male" ? 1 : 0,
//       smoke: req.body.smoke === "yes" ? 1 : 0,
//       years: parseFloat(req.body.years),
//       ldl: parseFloat(req.body.ldl),
//       chp: parseFloat(req.body.chp),
//       height: parseFloat(req.body.height),
//       weight: parseFloat(req.body.weight),
//       fh: req.body.fh === "yes" ? 1 : 0,
//       active: parseFloat(req.body.active),
//       lifestyle: parseFloat(req.body.lifestyle),
//       ihd: req.body.ihd === "yes" ? 1 : 0,
//       hr: parseFloat(req.body.hr),
//       dm: req.body.dm === "yes" ? 1 : 0,
//       bpsys: parseFloat(req.body.bpsys),
//       bpdias: parseFloat(req.body.bpdias),
//       htn: req.body.htn === "yes" ? 1 : 0,
//       ivsd: parseFloat(req.body.ivsd),
//       ecgpatt: parseFloat(req.body.ecgpatt),
//       qwave: parseFloat(req.body.qwave),
//     };

//     const userInputScaled = scaleData([Object.values(userInput)], scaler);

//     // Make predictions for user input
//     const predictions = model.predict(tf.tensor2d(userInputScaled));

//     // Convert predictions to 0 or 1
//     const predictedLabel = predictions.dataSync()[0] > 0.5 ? 1 : 0;

//     // Return success response with predicted label
//     res.json({ prediction: predictedLabel,suceess:true,result:"Heart diesease prediction succesfully done" });
//   } catch (error) {
//     console.error("Error training and saving the model", error);
//     res.status(500).json({ error: "Internal server error" });
//   }
// };

// module.exports = HeartDiseasePredictionController;
