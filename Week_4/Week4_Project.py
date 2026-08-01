import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

heart_data = pd.read_csv("heart.csv")

print("Heart Disease Dataset")
print(heart_data.head())

print()

print("Dataset Shape")
print(heart_data.shape)

print()

print("Column Names")
print(heart_data.columns)

print()

print("Missing Values")
print(heart_data.isnull().sum())

heart_data["trestbps"] = heart_data["trestbps"].fillna(heart_data["trestbps"].median())
heart_data["chol"] = heart_data["chol"].fillna(heart_data["chol"].median())
heart_data["thalch"] = heart_data["thalch"].fillna(heart_data["thalch"].median())
heart_data["oldpeak"] = heart_data["oldpeak"].fillna(heart_data["oldpeak"].median())

heart_data["ca"] = heart_data["ca"].fillna(heart_data["ca"].mode()[0])
heart_data["thal"] = heart_data["thal"].fillna(heart_data["thal"].mode()[0])
heart_data["slope"] = heart_data["slope"].fillna(heart_data["slope"].mode()[0])
heart_data["fbs"] = heart_data["fbs"].fillna(heart_data["fbs"].mode()[0])
heart_data["restecg"] = heart_data["restecg"].fillna(heart_data["restecg"].mode()[0])
heart_data["exang"] = heart_data["exang"].fillna(heart_data["exang"].mode()[0])

print()

print("Missing Values After Cleaning")
print(heart_data.isnull().sum())

sex_encoder = LabelEncoder()
heart_data["sex"] = sex_encoder.fit_transform(heart_data["sex"])

dataset_encoder = LabelEncoder()
heart_data["dataset"] = dataset_encoder.fit_transform(heart_data["dataset"])

cp_encoder = LabelEncoder()
heart_data["cp"] = cp_encoder.fit_transform(heart_data["cp"])

fbs_encoder = LabelEncoder()
heart_data["fbs"] = fbs_encoder.fit_transform(heart_data["fbs"])

restecg_encoder = LabelEncoder()
heart_data["restecg"] = restecg_encoder.fit_transform(heart_data["restecg"])

exang_encoder = LabelEncoder()
heart_data["exang"] = exang_encoder.fit_transform(heart_data["exang"])

slope_encoder = LabelEncoder()
heart_data["slope"] = slope_encoder.fit_transform(heart_data["slope"])

thal_encoder = LabelEncoder()
heart_data["thal"] = thal_encoder.fit_transform(heart_data["thal"])

for i in range(len(heart_data)):
    if heart_data.loc[i, "num"] == 0:
        heart_data.loc[i, "num"] = 0
    else:
        heart_data.loc[i, "num"] = 1

print()

print("First 5 Rows After Preprocessing")
print(heart_data.head())

X = heart_data.drop("id", axis=1)
X = X.drop("num", axis=1)

y = heart_data["num"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print()

print("Training Data Shape")
print(X_train.shape)

print()

print("Testing Data Shape")
print(X_test.shape)

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)

print()

print("Model Performance")

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.show()

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve")

plt.show()

new_patient = X_test[0].reshape(1, -1)

result = model.predict(new_patient)

print()

if result[0] == 1:
    print("Patient is likely to have Heart Disease.")
else:
    print("Patient is not likely to have Heart Disease.")
