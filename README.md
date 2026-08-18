# Machine Learning Assignment 2

## Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset and deploy the results using a Streamlit web application. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

---

## Dataset Description

Dataset Name: Breast Cancer Wisconsin Dataset

Source: Scikit-Learn

Number of Instances: 569

Number of Features: 30

Target Classes:
- 0 = Malignant
- 1 = Benign

This dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The goal is to classify tumors as malignant or benign.

---

## GitHub Repository Link



https://github.com/PradumSahu/MachineLearning_Assignment_2

---

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

---

## Evaluation Metrics Comparison

| Model               | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.9825 | 0.9954 |    0.9861 | 0.9861 |   0.9861 | 0.9623 |
| Decision Tree       |   0.9123 | 0.9157 |    0.9559 | 0.9028 |   0.9286 | 0.8174 |
| KNN                 |   0.9561 | 0.9788 |    0.9589 | 0.9722 |   0.9655 | 0.9054 |
| Naive Bayes         |   0.9386 | 0.9878 |    0.9452 | 0.9583 |   0.9517 | 0.8676 |
| Random Forest       |   0.9561 | 0.9931 |    0.9589 | 0.9722 |   0.9655 | 0.9054 |

---

## Model Performance Observations

| Model               | Observation                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| Logistic Regression | Achieved high accuracy and generalized well on unseen data.                            |
| Decision Tree       | Easy to interpret but showed slightly lower performance due to overfitting tendencies. |
| KNN                 | Performed well after feature scaling and handled local patterns effectively.           |
| Naive Bayes         | Fastest model but was slightly affected by feature dependency assumptions.             |
| Random Forest       | Produced the most balanced and robust performance across evaluation metrics.           |
| Overall Winner      | Logistic Regression achieved the best overall performance on the dataset.              |


---

## Project Structure

ML_Assignment_2/

├── app.py

├── train_models.py

├── requirements.txt

├── README.md

├── test_data.csv

└── model/

    ├── logistic_regression.pkl
    
    ├── decision_tree.pkl
    
    ├── knn.pkl
    
    ├── naive_bayes.pkl
    
    └── random_forest.pkl

---

## Running the Application

Install dependencies:

pip install -r requirements.txt

Train models:

python train_models.py

Launch Streamlit App:

streamlit run app.py

---

## Streamlit Features

- Upload Test CSV File
- Select Classification Model
- View Evaluation Metrics
- View Confusion Matrix
- View Classification Report