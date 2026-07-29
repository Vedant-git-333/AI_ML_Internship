import pandas as pd

data = pd.read_csv("Titanic-Dataset.csv")

print(data.isnull().sum())

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

print(data.isnull().sum())
