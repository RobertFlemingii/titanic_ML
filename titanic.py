import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Read the training data from a CSV file and store it in the train_data DataFrame.
train_data = pd.read_csv("train.csv")
print(train_data.head())  # Display the first few rows of the training data.

# Read the test data from a CSV file and store it in the test_data DataFrame.
test_data = pd.read_csv("test.csv")
print(test_data.head())  # Display the first few rows of the test data.

# Calculate the survival rate for women in the training data.
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

# Calculate the survival rate for men in the training data.
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

# Create the target variable 'y' by extracting the 'Survived' column from the training data.
y = train_data["Survived"]

# Define a list of features to use for training the model.
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Create the feature matrix 'X' by one-hot encoding the selected features in the training data.
X = pd.get_dummies(train_data[features])

# Create the feature matrix 'X_test' for the test data using the same features and one-hot encoding.
X_test = pd.get_dummies(test_data[features])

# Create a Random Forest Classifier model with specific parameters.
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train the model on the training data (X) with corresponding labels (y).
model.fit(X, y)

# Use the trained model to make predictions on the test data.
predictions = model.predict(X_test)

# Create a DataFrame with 'PassengerId' and 'Survived' columns for the submission.
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Save the submission DataFrame to a CSV file named 'submission.csv'.
output.to_csv('submission.csv', index=False)

# Display a success message.
print("Your submission was successfully saved!")
