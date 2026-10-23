
# Task 2: Supervised Machine Learning Models
# Breast Cancer Classification
# AI Pioneers ML Internship

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create output folder for results
os.makedirs("results", exist_ok=True)

# 1. Load the Breast Cancer dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

print("=" * 60)
print("BREAST CANCER CLASSIFICATION")
print("=" * 60)
print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("\nClass distribution:")
print(y.map({0: "Malignant", 1: "Benign"}).value_counts())

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Define supervised learning models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

# 4. Train and evaluate each model
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0)
    })

    print(f"{name} training and evaluation completed.")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.target_names,
        yticklabels=dataset.target_names
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_").replace("-", "")
    plt.savefig(f"results/{filename}_confusion_matrix.png", dpi=300)
    plt.close()

# 5. Compare all models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Save comparison table
results_df.to_csv("results/model_comparison.csv", index=False)

# 6. Plot model performance comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

results_plot = results_df.set_index("Model")[metrics]
results_plot.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))

plt.title("Performance Comparison of Supervised ML Models")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=300)
plt.close()

# 7. Display the best model based on F1-score
best_model = results_df.iloc[0]

print("\nBest model based on F1-Score:", best_model["Model"])
print(f"F1-Score: {best_model['F1-Score']:.4f}")
print("\nResults saved in the 'results' folder.")
print("Project execution completed successfully.")
