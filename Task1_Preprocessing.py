
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Create images folder if it does not exist
os.makedirs("images", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("titanic.csv", sep="\t")

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Duplicate Rows in Original Dataset ---")
print(df.duplicated().sum())


# 2. Handle Missing Values: Age
df["Age"] = df["Age"].fillna(df["Age"].median())


# 3. Handle Missing Values: Cabin
df["CabinKnown"] = df["Cabin"].notna().astype(int)


# 4. Handle Missing Values: Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())


# 5. Feature Selection
selected_features = df[
    ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]
].copy()

print("\n--- Selected Features ---")
print(selected_features.head())
print("Selected dataset shape:", selected_features.shape)


# 6. Encode Categorical Variables
encoded_df = pd.get_dummies(
    selected_features,
    columns=["Sex", "Embarked"],
    dtype=int
)

print("\n--- Encoded Dataset ---")
print(encoded_df.head())
print("Encoded columns:", encoded_df.columns.tolist())


# 7. Feature Scaling: Min-Max Normalization
scaler = MinMaxScaler()

encoded_df[["Age", "Fare"]] = scaler.fit_transform(
    encoded_df[["Age", "Fare"]]
)

print("\n--- Scaled Dataset ---")
print(encoded_df.head())
print("\nMinimum values:")
print(encoded_df[["Age", "Fare"]].min())
print("\nMaximum values:")
print(encoded_df[["Age", "Fare"]].max())


# 8. Exploratory Data Analysis: Survival Count
print("\n--- Survival Count ---")
print(df["Survived"].value_counts())

print("\n--- Survival Status ---")
print(
    df["Survived"].value_counts().rename({
        0: "Did Not Survive",
        1: "Survived"
    })
)


# GRAPH 1: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Titanic Passenger Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.tight_layout()
plt.savefig("images/survival_count.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# 9. Survival Analysis by Gender
survival_by_gender = df.groupby("Sex")["Survived"].sum()

print("\n--- Survivors by Gender ---")
print(survival_by_gender)

survival_rate = df.groupby("Sex")["Survived"].mean() * 100

print("\n--- Survival Rate by Gender (%) ---")
print(survival_rate.round(2))


# GRAPH 2: Survival Rate by Gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("images/survival_by_gender.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# 10. Survival Analysis by Passenger Class
survival_by_class = df.groupby("Pclass")["Survived"].sum()

print("\n--- Survivors by Passenger Class ---")
print(survival_by_class)


# GRAPH 3: Survival Rate by Passenger Class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Pclass", y="Survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("images/survival_by_class.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# 11. Correlation Analysis
numeric_df = df[
    ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
]

correlation = numeric_df.corr()

print("\n--- Correlation Matrix ---")
print(correlation.round(2))


# GRAPH 4: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Titanic Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# 12. Remove Duplicate Rows from Processed Features
print("\n--- Duplicate Rows Before Cleaning ---")
print(encoded_df.duplicated().sum())

encoded_df = encoded_df.drop_duplicates().copy()

print("\n--- Duplicate Rows After Cleaning ---")
print(encoded_df.duplicated().sum())

print("\n--- Final Dataset Shape ---")
print(encoded_df.shape)


# 13. Save Preprocessed Dataset
encoded_df.to_csv("preprocessed_titanic.csv", index=False)
print("\nPreprocessed dataset saved successfully!")


# 14. Final Dataset Validation
print("\n--- Final Dataset Information ---")
encoded_df.info()

print("\n--- Missing Values in Final Dataset ---")
print(encoded_df.isnull().sum())

print("\n--- Final Dataset Shape ---")
print(encoded_df.shape)


# 15. Preprocessing Summary
print("\n===== PREPROCESSING SUMMARY =====")
print("Original dataset rows:", len(df))
print("Final processed dataset rows:", len(encoded_df))
print("Final processed dataset columns:", len(encoded_df.columns))
print("Missing values in processed dataset:", encoded_df.isnull().sum().sum())
print("Duplicate rows in processed dataset:", encoded_df.duplicated().sum())
print("Final dataset saved as: preprocessed_titanic.csv")

print("\nAll 4 graph images are saved in the images folder:")
print("1. images/survival_count.png")
print("2. images/survival_by_gender.png")
print("3. images/survival_by_class.png")
print("4. images/correlation_heatmap.png")