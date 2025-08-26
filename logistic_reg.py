#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries
# Import the necessary libraries, including NumPy, pandas, scikit-learn, and matplotlib.

# In[19]:


# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")


# # Load Dataset
# Load the dataset using pandas.

# In[20]:


# Load the dataset using pandas
data = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
data.head()


# # Preprocess Data
# Handle missing values, encode categorical variables, and normalize the data if necessary.

# In[21]:


# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Normalize the data if necessary
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Class', axis=1))

# Create a DataFrame with the scaled features
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
scaled_data['Class'] = data['Class'].values

# Display the first few rows of the preprocessed dataset
scaled_data.head()


# # Split Dataset
# Split the dataset into training and testing sets using train_test_split from scikit-learn.

# In[22]:


# Split the dataset into training and testing sets
X = scaled_data.drop('Class', axis=1)
y = scaled_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Train Logistic Regression Model
# Use LogisticRegression from scikit-learn to train the model on the training data.

# In[23]:


# Train Logistic Regression Model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_regression_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
# Calculate and print the F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")
# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Evaluate Model
# Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score.

# In[9]:


# Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Make Predictions
# Use the trained model to make predictions on new data and visualize the results.

# In[17]:


# Make Predictions

# Prompt the user to input values for the new data point
print("Please enter the values for the new data point:")

# Example ranges for each feature (replace with actual ranges from your dataset)
# These ranges are just placeholders and should be replaced with the actual ranges from your dataset
ranges = {
    "Time": "0 to max time value",
    "V1": "min to max V1 value",
    "V2": "min to max V2 value",
    "V3": "min to max V3 value",
    "V4": "min to max V4 value",
    "V5": "min to max V5 value",
    "V6": "min to max V6 value",
    "V7": "min to max V7 value",
    "V8": "min to max V8 value",
    "V9": "min to max V9 value",
    "V10": "min to max V10 value",
    "V11": "min to max V11 value",
    "V12": "min to max V12 value",
    "V13": "min to max V13 value",
    "V14": "min to max V14 value",
    "V15": "min to max V15 value",
    "V16": "min to max V16 value",
    "V17": "min to max V17 value",
    "V18": "min to max V18 value",
    "V19": "min to max V19 value",
    "V20": "min to max V20 value",
    "V21": "min to max V21 value",
    "V22": "min to max V22 value",
    "V23": "min to max V23 value",
    "V24": "min to max V24 value",
    "V25": "min to max V25 value",
    "V26": "min to max V26 value",
    "V27": "min to max V27 value",
    "V28": "min to max V28 value",
    "Amount": "0 to max amount value"
}

# Collect user input for each feature
new_data = []
for feature, range_info in ranges.items():
    value = float(input(f"Enter value for {feature} ({range_info}): "))
    new_data.append(value)

# Convert the new data to a numpy array
new_data = np.array([new_data])

new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = logistic_regression_model.predict(new_data_scaled)

# Map predictions to labels
prediction_labels = ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]

# Print the predictions
print("Predictions for the new data:", prediction_labels)

# Visualize the results
plt.figure(figsize=(10, 7))
plt.scatter(range(len(prediction_labels)), prediction_labels, color='blue', label='Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Class')
plt.title('Predictions on New Data')
plt.legend()
plt.show()

