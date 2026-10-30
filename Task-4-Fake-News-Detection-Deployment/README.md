
# Fake News Detection System

## Project Overview
This project is an end-to-end Machine Learning application that classifies news articles as Fake News or Real News using Natural Language Processing (NLP).

Users can enter a news article through a Flask web interface or send text to a REST API to receive a prediction.

## Objectives
- Build an end-to-end machine learning project.
- Apply NLP and TF-IDF for text feature extraction.
- Train and evaluate a Logistic Regression classifier.
- Save the trained model using Joblib.
- Deploy the model through a Flask web application and prediction API.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Flask
- Joblib
- HTML and CSS

## Dataset
- Dataset: WELFake Dataset
- Original records: 72,134
- Cleaned records: 63,637
- Input feature: News content
- Target: Label
- Label 0: Fake News
- Label 1: Real News

## Machine Learning Workflow
1. Dataset loading
2. Data cleaning and preprocessing
3. Combining news title and article text
4. Removing missing and duplicate records
5. Train-test splitting (80:20)
6. TF-IDF feature extraction
7. Logistic Regression model training
8. Model evaluation
9. Model serialization using Joblib
10. Flask application deployment

## Model Performance

Test Accuracy: 95.15%

| Metric | Fake News | Real News |
|---|---:|---:|
| Precision | 0.95 | 0.95 |
| Recall | 0.96 | 0.94 |
| F1-score | 0.96 | 0.95 |

## Project Structure

```text
Task-4-Fake-News-Detection-Deployment/
│
├── dataset/
│   ├── WELFake_Dataset.csv
│   └── cleaned_news.csv
│
├── templates/
│   └── index.html
│
├── train_model.py
├── app.py
├── fake_news_model.joblib
├── requirements.txt
├── requirements-full.txt
└── README.md
```

## Installation

Clone the repository or open the project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

## Run the Application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

## Prediction API

Endpoint:

`POST /predict`

Example JSON request:

```json
{
  "news": "Enter your news article here."
}
```

Example response:

```json
{
  "prediction": "Fake News",
  "label": 0
}
```

The actual prediction depends on the input article.

## Limitations
- Predictions depend on patterns learned from the training dataset.
- The model may misclassify some articles.
- The model does not independently verify facts or sources.
- Predictions should not be treated as definitive proof that a news article is fake or real.

## Conclusion
This project demonstrates an end-to-end NLP-based machine learning workflow, from data preprocessing and model training to model serialization and deployment using Flask.

It provides a practical example of applying Machine Learning to text classification.
