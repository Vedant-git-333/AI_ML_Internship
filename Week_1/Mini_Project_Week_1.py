import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = pd.read_csv("Titanic-Dataset.csv")

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])

ohe = OneHotEncoder(sparse_output=False)
new = ohe.fit_transform(data[["Embarked"]])
new = pd.DataFrame(new, columns=ohe.get_feature_names_out(["Embarked"]))

data = pd.concat([data, new], axis=1)

plt.hist(data["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("age_distribution.png")
plt.show()

data.to_csv("cleaned_titanic.csv", index=False)

print("Project Completed")
