import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = pd.read_csv("Titanic-Dataset.csv")

data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

ohe = OneHotEncoder(sparse_output=False)
new = ohe.fit_transform(data[["Embarked"]])
new = pd.DataFrame(new, columns=ohe.get_feature_names_out(["Embarked"]))

data = pd.concat([data, new], axis=1)

print(data.head())
