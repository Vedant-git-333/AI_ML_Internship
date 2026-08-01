import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

iris_dataset = pd.read_csv("Iris.csv")

flower_features = iris_dataset.iloc[:, 1:-1]
species_codes = iris_dataset["Species"].astype("category").cat.codes

pca_model = PCA(n_components=2)

reduced_features = pca_model.fit_transform(flower_features)

print("Original Shape:", flower_features.shape)
print("Reduced Shape:", reduced_features.shape)

plt.figure(figsize=(8, 6))

plt.scatter(
    reduced_features[:, 0],
    reduced_features[:, 1],
    c=species_codes,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")

plt.show()
