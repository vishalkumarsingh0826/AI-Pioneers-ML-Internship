# Titanic Dataset — Python for Machine Learning & Data Preprocessing

## 📌 Project Overview

This project focuses on data exploration, cleaning, preprocessing, feature encoding, feature scaling, and exploratory data analysis (EDA) using Python. The Titanic passenger dataset is used to understand the factors associated with passenger survival.

**Internship:** AI Pioneers — Machine Learning Internship
**Task:** Task 1 — Python for Machine Learning & Data Preprocessing

## 🎯 Objectives

* Load and explore a dataset using Pandas and NumPy.
* Identify and handle missing values.
* Create a new feature from existing data.
* Select relevant features for machine learning.
* Encode categorical variables into numerical values.
* Apply Min-Max scaling to numerical features.
* Perform exploratory data analysis using visualizations.
* Export the preprocessed dataset to a CSV file.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 📂 Project Structure

```text
Task_1_Python_Data_Perprocessing/
│
├── images/
│   ├── survival_count.png
│   ├── survival_by_gender.png
│   ├── survival_by_class.png
│   └── correlation_heatmap.png
│
├── titanic.csv
├── preprocessed_titanic.csv
├── Task1_Preprocessing.py
└── README.md
```

## 🔍 Work Completed

### 1. Data Loading and Exploration

* Loaded the Titanic dataset using Pandas.
* Inspected the first five rows and dataset information.
* Checked missing values, duplicate records, and statistical summaries.

### 2. Data Cleaning

* Filled missing `Age` values using the median.
* Filled missing `Embarked` values using the mode.
* Created a `CabinKnown` feature to indicate whether cabin information was available.

### 3. Feature Selection

Selected the following features for preprocessing:

* `Survived`
* `Pclass`
* `Sex`
* `Age`
* `Fare`
* `Embarked`

### 4. Categorical Encoding

Applied one-hot encoding to the `Sex` and `Embarked` columns using Pandas `get_dummies()`.

### 5. Feature Scaling

Applied Scikit-learn's `MinMaxScaler` to the `Age` and `Fare` features to scale their values between 0 and 1.

### 6. Exploratory Data Analysis (EDA)

Created four visualizations to explore survival patterns and numerical relationships in the dataset.

## 📊 Graphs and Visualizations

### 1. Titanic Passenger Survival Count

This chart compares the number of passengers who survived with those who did not survive.

![Titanic Passenger Survival Count](images/survival_count.png)

### 2. Survival Rate by Gender

This chart compares the survival rate of female and male passengers.

![Survival Rate by Gender](images/survival_by_gender.png)

### 3. Survival Rate by Passenger Class

This chart compares survival rates across passenger classes.

![Survival Rate by Passenger Class](images/survival_by_class.png)

### 4. Correlation Heatmap

This heatmap displays correlations between selected numerical features, including survival, passenger class, age, family-related variables, and fare.

![Titanic Dataset Correlation Heatmap](images/correlation_heatmap.png)

## 💾 Output Files

* **`Task1_Preprocessing.py`** — Python script containing the preprocessing and visualization workflow.
* **`preprocessed_titanic.csv`** — Exported dataset after feature selection, encoding, scaling, and duplicate removal.
* **`images/`** — Folder containing the four generated graph images.

## ▶️ How to Run the Project

1. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Keep `titanic.csv` in the project folder.

3. Ensure the dataset is tab-separated, as expected by the script.

4. Run the Python script:

   ```bash
   python Task1_Preprocessing.py
   ```

5. Check the `images/` folder for the generated visualizations and the project folder for `preprocessed_titanic.csv`.

## 📚 Key Learnings

* Data inspection and cleaning using Pandas.
* Handling missing values using statistical methods.
* Feature engineering and feature selection.
* Converting categorical data into numerical features.
* Scaling numerical features for machine learning.
* Visualizing data distributions and relationships.
* Exporting processed data for downstream analysis.



## ✅ Conclusion

This project provided practical experience with the essential stages of a machine learning preprocessing workflow. The dataset was explored, selected missing values were handled, categorical features were encoded, numerical features were scaled, and visualizations were generated to investigate passenger survival patterns.

The resulting CSV file can serve as a starting point for further analysis or machine learning experiments, subject to appropriate validation of the data and preprocessing choices.



*Completed as part of the AI Pioneers Machine Learning Internship — Task 1.*
