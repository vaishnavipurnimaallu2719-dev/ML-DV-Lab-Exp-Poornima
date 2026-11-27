import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    random_state=42
)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()