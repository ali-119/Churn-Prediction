# Churn Prediction Project

# Overview
This project focuses on predicting **customer churn** — determining whether a customer will leave a company’s service or not.  
Such predictive modeling helps businesses (especially in telecom, banking, and SaaS industries) to **identify at-risk customers** and take early retention actions.

-----

# Dataset
- **Name:** Telco Customer Churn Dataset  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Size:** 7032 records and 21 features  
- **Content:** Customer information, service usage, contract types, payment details, and churn status.

-----

# Project Workflow
## Data Preparation
- Inspected and handled missing values  
- Removed irrelevant columns (`customerID`)  
- Converted categorical variables using `LabelEncoder`  
- Scaled numerical features as needed  

## Exploratory Data Analysis (EDA)
- Visualized distributions (`countplot`, `boxplot`, `violinplot`)  
- Checked correlations between variables (`corr()`)  
- Explored patterns between churn and service types  

## Model Training
The following **classification** models were built and evaluated:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

Hyperparameter tuning was applied using **GridSearchCV** for Decision Tree (as an example).
For the rest of the models, the default model mode of the models themselves was used.

## Model Evaluation
The models were compared using multiple performance metrics:
  
**For numerical analyses**
- classification_report which includes:  
-- Accuracy
-- Precision
-- Recall
-- F1-score
  
**To analyze numbers in the form of a Display chart**
- ConfusionMatrixDisplay

A summary table and visual comparison were created to display the results clearly.

-----

## Results
The **Gradient Boosting** and **Adaboost** classifiers achieved the highest accuracy.
Among them, **Adaboost** achieved an accuracy of approximately **81%** on the test set.

**Top influential features:**
- Contract Type  
- TechSupport Availability  
- Customer Tenure  

-----

# Libraries Used
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib
- warnings

-----

# How to Run
## Clone the repository:
git clone [github](https://github.com/ali-119/Churn-Prediction)
cd Churn_Prediction

## Install dependencies:
<pre> pip install -r requirements.txt </pre>

## Open the Jupyter notebook:
jupyter notebook notebooks/Churn_Prediction.ipynb

Run all cells to train and evaluate the model.

-----

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)
