import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris_dataset = pd.read_csv("Iris.csv")

flower_features = iris_dataset.iloc[:, :-1]

true_labels = iris_dataset.iloc[:, -1]

kmeans_model = KMeans(n_clusters=3, random_state=42)

predicted_clusters = kmeans_model.fit_predict(flower_features)

comparison = pd.DataFrame({
    "True Label": true_labels,
    "Predicted Cluster": predicted_clusters
})

print(comparison)

pca_model = PCA(n_components=2)

reduced_features = pca_model.fit_transform(flower_features)

plt.figure(figsize=(8,6))

plt.scatter(
    reduced_features[:,0],
    reduced_features[:,1],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Flower Clustering using K-Means")

plt.show()
