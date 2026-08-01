import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris_dataset = pd.read_csv("Iris.csv")

flower_features = iris_dataset.iloc[:, 1:-1]

kmeans_model = KMeans(n_clusters=3, random_state=42)

predicted_clusters = kmeans_model.fit_predict(flower_features)

print("Cluster assigned to each flower:")
print(predicted_clusters)

plt.figure(figsize=(8, 6))

plt.scatter(
    flower_features.iloc[:, 0],
    flower_features.iloc[:, 1],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

cluster_centers = kmeans_model.cluster_centers_

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()

plt.show()
