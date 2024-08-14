import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib


# Load the dataset
data = pd.read_csv("D:\DoctorAppointmentSystem\server\myenv\ECG-Dataset.csv")

# Assuming the target column is named 'target'
X = data.drop(columns=["target"])
y = data["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initialize models
models = {
    "random_forest": RandomForestClassifier(),
    "svm": SVC(),
    "knn": KNeighborsClassifier(),
}

# Train and save the models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test, "X_test_scaled.pkl")
joblib.dump(y_test, "y_test.pkl")
