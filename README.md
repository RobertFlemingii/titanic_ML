# Titanic Survival Prediction Model
# Overview
A machine learning model to predict passenger survival on the Titanic using Random Forest Classification.
# Setup
# Virtual Environment

Create virtual environment

python -m venv env

Activate virtual environment
On Unix/macOS:

source env/bin/activate

On Windows:

env\Scripts\activate

# Dependencies
pip install numpy pandas scikit-learn

# Project Structure
# Data Files

train.csv - Training dataset with known survival outcomes
test.csv - Test dataset for predictions
gender_submission.csv - Example submission format
submission.csv - Output file with model predictions

# Features

Pclass - Passenger class (1st, 2nd, 3rd)
Sex - Passenger's gender
SibSp - Number of siblings/spouses aboard
Parch - Number of parents/children aboard

# Model Information
# Classifier Details

RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=1
)

# Code Structure
1. Data Loading and Inspection

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

2. Initial Analysis

Calculate survival rates by gender
Provide baseline insights

3. Feature Engineering

Feature selection
One-hot encoding

4. Model Training and Prediction

Model training
Generate predictions

5. Output Generation

Create submission file
Save predictions

# Usage Instructions

Activate virtual environment
Ensure all CSV files are present:

train.csv
test.csv
gender_submission.csv


Run the script
Check 'submission.csv' for predictions

# Output Format
# submission.csv structure

Column         Description
PassengerId    Passenger identification number
Survived       Predicted survival (0 = No, 1 = Yes)
