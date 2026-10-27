# Task 3: Unsupervised Learning and Model Evaluation

## Project: Customer Segmentation Using Clustering

### Overview
This project demonstrates unsupervised learning using K-Means and Hierarchical Clustering. A sample dataset containing 20 customers is used, with two features: annual income and spending score.

### Objectives
- Understand unsupervised learning and clustering.
- Apply K-Means and Hierarchical Clustering.
- Use the Elbow Method to examine the appropriate number of clusters.
- Evaluate clustering with the Silhouette Score.
- Explore dimensionality reduction using Principal Component Analysis (PCA).

### Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

### Dataset
The dataset is generated for educational demonstration and contains 20 sample customer records. It is not collected real-world customer data.

### Algorithms
1. **K-Means Clustering:** Groups customers into three clusters based on standardized income and spending-score features.
2. **Hierarchical Clustering:** Creates three clusters using agglomerative clustering.
3. **PCA:** Examines the principal components of the two standardized features.

### Results

| Metric | K-Means | Hierarchical |
|---|---:|---:|
| Silhouette Score | 0.5866 | 0.5866 |

Both algorithms achieved the same Silhouette Score on this sample dataset.

PCA retained two components from the two original features, explaining 100% of the total variance. Since the input already had two features and PCA also retained two, dimensionality was not reduced.

### Generated Files
- `customer_data.csv` — sample dataset
- `clustering_results.csv` — customer records with cluster labels
- `elbow_method.png` — Elbow Method plot
- `kmeans_clusters.png` — K-Means visualization
- `hierarchical_clusters.png` — Hierarchical Clustering visualization
- `create_dataset.py` — dataset creation script
- `clustering_analysis.py` — clustering, evaluation, and PCA script

### How to Run
Run these commands in the project directory:

```bash
python create_dataset.py
python clustering_analysis.py
```

### Conclusion
This project demonstrates two unsupervised clustering algorithms and evaluates their cluster quality using the Silhouette Score. The visualizations and Elbow Method support cluster analysis, while PCA introduces dimensionality reduction concepts. The results apply to this small, artificially generated dataset and should not be generalized to real customers.
