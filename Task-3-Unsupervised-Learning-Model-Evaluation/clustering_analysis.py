
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("customer_data.csv")

# Select features for clustering
X = df[["Annual_Income", "Spending_Score"]]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the best number of clusters using the Elbow Method
inertias = []
k_values = range(2, 8)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Apply Hierarchical Clustering with 3 clusters
hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

# Evaluate clustering quality
kmeans_score = silhouette_score(X_scaled, df["KMeans_Cluster"])
hierarchical_score = silhouette_score(
    X_scaled, df["Hierarchical_Cluster"]
)

print("K-Means Silhouette Score:", round(kmeans_score, 4))
print("Hierarchical Silhouette Score:", round(hierarchical_score, 4))

# Visualize K-Means clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["KMeans_Cluster"],
    cmap="viridis",
    s=80
)
plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (sample units)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.close()

# Visualize Hierarchical clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Hierarchical_Cluster"],
    cmap="plasma",
    s=80
)
plt.title("Customer Segmentation Using Hierarchical Clustering")
plt.xlabel("Annual Income (sample units)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.savefig("hierarchical_clusters.png")
plt.close()

# Save results
df.to_csv("clustering_results.csv", index=False)

print("\nAnalysis completed successfully!")
print("Files created:")
print("- elbow_method.png")
print("- kmeans_clusters.png")
print("- hierarchical_clusters.png")
print("- clustering_results.csv")

from sklearn.decomposition import PCA

# Reduce the standardized data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Analysis")
print("Original number of features:", X_scaled.shape[1])
print("Reduced number of features:", X_pca.shape[1])
print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_.round(4)
)
print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)