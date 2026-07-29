import pandas as pd

data = pd.read_csv("Titanic-Dataset.csv")

print(data.head())
print(data.info())
print(data.describe())
