import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from google.colab import drive
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify

app = Flask(__name__)

dataset = pd.read_csv("heart.csv")
# Read a comma-separated values (csv) file into DataFrame. Also supports optionally iterating or breaking the file into chunks.


##Understanding the coloum better
info = [
    "age",
    "1: male, 0: female",
    "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
    "resting blood pressure",
    " serum cholestoral in mg/dl",
    "fasting blood sugar > 120 mg/dl",
    "resting electrocardiographic results (values 0,1,2)",
    " maximum heart rate achieved",
    "exercise induced angina",
    "oldpeak = ST depression induced by exercise relative to rest",
    "the slope of the peak exercise ST segment",
    "number of major vessels (0-3) colored by flourosopy",
    "thal: 1 = normal; 2 = fixed defect; 3 = reversable defect",
]
for i in range(len(info)):
    print(dataset.columns[i] + ":\t\t\t" + info[i])

###### Checking correlation between columns

print(dataset.corr()["target"].abs().sort_values(ascending=False))
##This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.

#### Exploratory Data Analysis (EDA)

### First, analysing the target variable:
y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)

print(
    "Percentage of patience without heart problems: "
    + str(round(target_temp[0] * 100 / 72184, 2))
)
print(
    "Percentage of patience with heart problems: "
    + str(round(target_temp[1] * 100 / 72184, 2))
)


def onehot(ser, num_classes=None):
    """
    One-hot encode the series.
    Example:
    >>> onehot([1, 0, 2], 3)
    array([[0., 1., 0.],
       [1., 0., 0.],
       [0., 0., 1.]])
    """
    if num_classes == None:
        num_classes = len(np.unique(ser))
    return np.identity(num_classes)[ser]


new_col_names = []
need_encode_col = ["restecg", "thal", "slope", "cp"]
no_encode_col = [col for col in dataset.columns if col not in need_encode_col]
new_dataset = dataset[no_encode_col]
for col in need_encode_col:
    num_classes = len(dataset[col].unique())
    new_col_names = [f"{col}_{i}" for i in range(num_classes)]
    encoded = pd.DataFrame(
        onehot(dataset[col], num_classes), columns=new_col_names, dtype=int
    )
    new_dataset = pd.concat([new_dataset, encoded], axis=1)
new_dataset.head()


# The attributes including age, sex,chest pain type (4 values),resting blood pressure, serum cholestoral , fasting blood sugar ,
# resting electrocardiographic results ,maximum heart rate achieved, exercise induced angina,oldpeak = ST depression induced by exercise relative to rest,the slope of the peak exercise ST segment,
# number of major vessels (0-3) colored by flourosopy were used to define train and test the model.

data_cols = [col for col in new_dataset.columns if col != "target"]

print(data_cols)

X = new_dataset.drop(["target"], axis="columns")
X.head(3)
X.shape

X.columns


new_dataset_shfl = shuffle(new_dataset, random_state=443)
# define X and y
X = new_dataset_shfl[data_cols].values
y = new_dataset_shfl["target"].values
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=80
)

# Confusion matrix

# Table that describes the performance of a classification model Every observation in the testing set is represented in exactly one box It's a 2x2 matrix because there are 2 response classes

# Basic terminology

# True Positives (TP): we correctly predicted that they do have heart diease
# True Negatives (TN): we correctly predicted that they don't have heart diease
# False Positives (FP): we incorrectly predicted that they do have heart diease(a "Type I error")
# False Negatives (FN): we incorrectly predicted that they don't have heart diease (a "Type II error")

# train KNeighborsClassifier model on the training set

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred_knn = classifier.predict(X_test)

# make class predictions for the testing set
knn_probs = classifier.predict_proba(X_test)
knn_probs = knn_probs[
    :, 1
]  #  If k value is set to 1, then you will get 0 or 1. If it's set to 2, you will get 0, 0.5 or 1.

result = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:- KNN")
print(result)

"""

Classification Accuracy: Overall, how often is the classifier correct?


* Accuracy : number of correctly predicted data points out of all the data points
* Precision : quantifies the number of positive class predictions that actually belong to the positive class.
* Recall : quantifies the number of positive class predictions made out of all positive examples in the dataset.
* F-Measure : provides a single score that balances both the concerns of precision and recall in one number.
"""

# Classification accuracy for KNN
result_classification_report_knn = classification_report(y_test, y_pred_knn)
print(
    " k nearest neighbor",
)
print(result_classification_report_knn)
accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy_score_knn)

# AUC For Knn
# calculate scores
knn_auc = roc_auc_score(y_test, knn_probs)
# summarize scores
print("k nearest neighbor: ROC AUC=%.3f" % (knn_auc))
# calculate roc curves
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
# plot the roc curve for the model
pyplot.plot(knn_fpr, knn_tpr, marker=".", label="k nearest neighbor")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# SVM model
# Import svm model


# Create a svm Classifier
clf = svm.SVC(kernel="linear")  # Linear Kernel
clf = svm.SVC(probability=True)
# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_svm = clf.predict(X_test)

svm_probs = clf.predict_proba(X_test)
svm_probs = svm_probs[:, 1]

# Classification accuracy for SVM
result = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:- Support Vector Machine")
print(result)
result_classification_report_svm = classification_report(y_test, y_pred_svm)

print(
    " Support Vector Machine",
)
print(result_classification_report_svm)
accuracy_score_vm = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy_score_vm)

# AUC of SVM
# calculate scores
svm_auc = roc_auc_score(y_test, svm_probs)
# summarize scores
print("Support Vector Machine: ROC AUC=%.3f" % (svm_auc))
# calculate roc curves
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
# plot the roc curve for the model
pyplot.plot(svm_fpr, svm_tpr, linestyle="--", label="Support Vector Machine")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Comparison the performance of knn and svm algorithm by AUC
# calculate scores
svm_auc = roc_auc_score(y_test, svm_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
# summarize scores
print("Support Vector Machine: ROC AUC=%.3f" % (svm_auc))
print("k nearest neighbor: ROC AUC=%.3f" % (knn_auc))
# calculate roc curves
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
# plot the roc curve for the model
pyplot.plot(svm_fpr, svm_tpr, linestyle="--", label="Support Vector Machine")
pyplot.plot(knn_fpr, knn_tpr, marker=".", label="k nearest neighbor")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Random Forest Algorithm
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)
y_pred_rf = classifier.predict(X_test)


rf_probs = classifier.predict_proba(X_test)
rf_probs = rf_probs[:, 1]

# Classification accuracy for Random Forest Algorithm
result = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:-Random Forest Algorithm")
print(result)
result_classification_report_randomForest = classification_report(y_test, y_pred_rf)

print(
    "Random Forest Algorithm",
)
# print(result_classification_report_randomForest)
print("Accuracy:", result4)

# performance of Random Forest by AUC
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
# summarize scores
print("Random Forest: ROC AUC=%.3f" % (rf_auc))
# calculate roc curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
# plot the roc curve for the model
pyplot.plot(rf_fpr, rf_tpr, marker=".", label="Random Forest")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Comparison of KNN and RNN
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
# summarize scores
print("Random Forest: ROC AUC=%.3f" % (rf_auc))
print("k nearest neighbor: ROC AUC=%.3f" % (knn_auc))
# calculate roc curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
# plot the roc curve for the model
pyplot.plot(rf_fpr, rf_tpr, marker=".", label="Random Forest")
pyplot.plot(knn_fpr, knn_tpr, marker=".", label="k nearest neighbor")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# Comparison of SVM and Rf
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
# summarize scores
print("Random Forest: ROC AUC=%.3f" % (rf_auc))
print("Support Vector Machine: ROC AUC=%.3f" % (svm_auc))
# calculate roc curves
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
# plot the roc curve for the model
pyplot.plot(rf_fpr, rf_tpr, marker=".", label="Random Forest")
pyplot.plot(svm_fpr, svm_tpr, linestyle="--", label="Support Vector Machine")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


nn = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=1
)

nn.fit(X_train, y_train)

y_pred_nn = nn.predict(X_test)

nn_probs = nn.predict_proba(X_test)
nn_probs = nn_probs[:, 1]

result = confusion_matrix(y_test, y_pred_nn)
print("Confusion Matrix:")
print(result)
result_cp = classification_report(y_test, y_pred_nn)
print(
    "Classification Report:",
)
# print(result1)
accuracy_score_random_forest = accuracy_score(y_test, y_pred_nn)
print("Accuracy:", accuracy_score_random_forest)

# compare the performance of five classifiers by AUC
# calculate scores
rf_auc = roc_auc_score(y_test, rf_probs)
svm_auc = roc_auc_score(y_test, svm_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

print("Support Vector Machine: ROC AUC=%.3f" % (svm_auc))
print("k nearest neighbor: ROC AUC=%.3f" % (knn_auc))
print("Random Forest: ROC AUC=%.3f" % (rf_auc))
# calculate roc curves

# plot the roc curve for the model
pyplot.plot(rf_fpr, rf_tpr, marker=".", label="Random Forest")
pyplot.plot(svm_fpr, svm_tpr, linestyle="--", label="Support Vector Machine")
pyplot.plot(knn_fpr, knn_tpr, marker=".", label="k nearest neighbor")
# axis labels
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


#  predicting the heart disease function


def predict_heart_diease(data):
    age = data["age"]
    sex = data["sex"]
    trestbps = data["trestbps"]
    chol = data["chol"]
    fbs = data["fbs"]
    thalach = data["thalach"]
    exang = data["exang"]
    oldpeak = data["oldpeak"]
    ca = data["ca"]
    restecg = data["restecg"]
    thal = data["thal"]
    slope = data["slope"]
    cp = data["cp"]

    X = new_dataset.drop(["target"], axis="columns")
    X.shape
    restecg_index = np.where(X.columns == restecg)[0][0]
    loc_thal = np.where(X.columns == thal)[0][0]
    loc_slope = np.where(X.columns == slope)[0][0]
    loc_cp = np.where(X.columns == cp)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = age
    x[1] = sex
    x[2] = trestbps
    x[3] = chol
    x[4] = fbs
    x[5] = thalach
    x[6] = exang
    x[7] = oldpeak
    x[8] = ca

    if restecg_index >= 0:
        x[restecg_index] = 1

    if loc_thal >= 0:
        x[loc_thal] = 1

    if loc_slope >= 0:
        x[loc_slope] = 1

    if loc_cp >= 0:
        x[loc_cp] = 1

    return nn.predict([x])[0]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = predict_heart_diease(data)
    return jsonify({"prediction": result})


# predict_heart_diease(
# 63, 1, 145, 233, 1, 150, 0, 2.3, 0, "restecg_0", "thal_1", "slope_0", "cp_3"
# )

# Export the tested model to a pickle file

with open("heart2.sav", "wb") as f:
    pickle.dump(nn, f)

# Export column information to a file that will be useful later on in our prediction application
X = new_dataset.drop(["target"], axis="columns")
X.shape

columns = {"data_columns": [col.lower() for col in X.columns]}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

corr = dataset.corr()
plt.subplots(figsize=(15, 10))
sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns,
    annot=True,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
)
sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns,
    annot=True,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
)

scores = [svm_auc, knn_auc, rf_auc]
algorithms = ["Support Vector Machine", "K-Nearest Neighbors", "Random Forest"]

for i in range(len(algorithms)):
    print(
        "The accuracy score achieved using "
        + algorithms[i]
        + " is: "
        + str(scores[i])
        + " %"
    )

sns.set(rc={"figure.figsize": (15, 8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms)

if __name__ == "__main__":
    app.run(debug=True)
