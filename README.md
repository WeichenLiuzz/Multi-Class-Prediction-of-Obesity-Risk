# Multi-Class Prediction of Obesity Risk
This project focuses on predicting the risk of obesity using machine learning techniques. The dataset used in this project contains various features related to lifestyle, diet, and physical activity, and the task is to predict the level of obesity risk based on these features.

## Dataset
ObesityDataSet.csv: This file contains the dataset used for training and evaluation. It includes features such as age, height, weight, family history of overweight, diet habits, physical activity level, etc.

## Files
train.csv: CSV file containing the training data.\
test.csv: CSV file containing the test data.\
sample_submission.csv: Sample submission file for the competition.

## Notebook
Multi-Class Prediction of Obesity Risk.ipynb: This Jupyter notebook contains the code for data preprocessing, exploratory data analysis, model training, and evaluation. It walks through the process of loading the data, performing feature engineering, training machine learning models, and evaluating their performance.

## Project Structure
Data Preprocessing: Handling missing values, encoding categorical variables, feature scaling, etc.\
Exploratory Data Analysis (EDA): Analyzing the distribution of features, exploring relationships between variables, identifying patterns, and gaining insights into the data.\
Model Training: Implementing machine learning models such as LightGBM, RandomForest, etc., for multi-class classification of obesity risk.\
Model Evaluation: Evaluating the models using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, etc.\
Hyperparameter Tuning: Optimizing model hyperparameters using techniques like RandomizedSearchCV, GridSearchCV, etc.\
Submission: Generating predictions on the test data and creating the submission file.

## Setup
To run the notebook and reproduce the results, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies specified in the requirements.txt file.
3. Run the Jupyter notebook Multi-Class Prediction of Obesity Risk.ipynb in your preferred environment.

## Dependencies
Python 3.x
Jupyter Notebook
pandas
NumPy
scikit-learn
LightGBM
