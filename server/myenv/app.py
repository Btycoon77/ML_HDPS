#  logistic Regression algorithm is used here

from flask import Flask, request, jsonify
import joblib
import pandas as pd   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
heart_data = pd.read_csv("ECG-Dataset.csv")

# Split data into features and target
X = heart_data.drop("target", axis=1)  # it means excluding the feature target ,it will include all other features
y = heart_data["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

app = Flask(__name__)

# Load the trained model and scaler 
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request
    user_data = request.get_json()

    # Preprocess user input
    user_data = pd.DataFrame([user_data])
    user_data_scaled = scaler.transform(user_data.drop("target", axis=1))

    # Make the prediction
    prediction = model.predict(user_data_scaled)

    # Convert the prediction to a response JSON
    response = {
        "prediction": int(prediction[0]),
        "success":"true",
        "result":"Succesfully predicted "
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port = 8000,debug=True);

#  to run flask application python app.py

