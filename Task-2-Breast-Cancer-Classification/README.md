
# Task 2: Supervised Machine Learning Models
## Comparative Analysis of Breast Cancer Classification

### Project Overview
This project implements and compares four supervised machine learning algorithms to classify breast cancer tumors as malignant or benign. The project uses the Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn.

The objective is to train different classification models, evaluate their performance using standard metrics, and identify the best-performing model on the test dataset.

### Dataset
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Scikit-learn built-in dataset
- **Total samples:** 569
- **Total features:** 30
- **Classes:** Malignant and Benign
- **Training samples:** 455
- **Testing samples:** 114
- **Test size:** 20%

No separate dataset download is required.

### Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

### Technologies Used
- Python 3.13.7
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Methodology
1. Load the built-in breast cancer dataset.
2. Separate the input features and target labels.
3. Split the dataset into training and testing sets.
4. Standardize features for Logistic Regression and KNN.
5. Train all four classification models.
6. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
7. Compare model performance and visualize the results.

### Model Performance

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 98.25% | 98.61% |
| Random Forest | 95.61% | 96.55% |
| K-Nearest Neighbors | 95.61% | 96.55% |
| Decision Tree | 91.23% | 92.86% |

Logistic Regression achieved the highest accuracy and F1-score on this particular test split.

### Project Structure

```text
Task-2-Breast-Cancer-Classification/
├── Task2_Breast_Cancer_Classification.py
├── README.md
└── results/
    ├── logistic_regression_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    ├── random_forest_confusion_matrix.png
    ├── knearest_neighbors_confusion_matrix.png
    ├── model_comparison.csv
    └── model_comparison.png
```

### How to Run

Install the required libraries if they are not already installed:

```bash
python -m pip install pandas matplotlib seaborn scikit-learn
```

Run the Python script:

```bash
python Task2_Breast_Cancer_Classification.py
```

The program displays classification reports and a model comparison table. It also saves the evaluation results and graphs in the `results` directory.

### Conclusion
This project demonstrates how supervised machine learning algorithms can be applied to a binary classification problem. Among the four tested models, Logistic Regression achieved the best performance, with 98.25% accuracy and a 98.61% F1-score on the selected test set. The experiment highlights the importance of evaluating and comparing multiple models using appropriate performance metrics.


