import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import losses
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Tensorflow Version"+ tf.__version__)
print("$$$ LOADING DATA USING CSV DATASET $$$")
path = Path(__file__).parent / r'D:\aRA\9XAIforDF\Dataset2\OneDrive20240213\CSVFiles\TrainingandTestingSets\UNSWNB15trainingset.csv'
training = pd.read_csv(path)
path2 = Path(__file__).parent / r'D:\aRA\9XAIforDF\Dataset2\OneDrive20240213\CSVFiles\TrainingandTestingSets\UNSWNB15testingset.csv'
testing = pd.read_csv(path2)
print("loading done...")

print("Preprocessing")
# Combine train and test data for preprocessing
combined_data = pd.concat([training, testing], axis=0)

# Separate features and target
X = combined_data.drop(columns=["attack_cat", "label"])
y = combined_data["attack_cat"]

# Encode target variable into numerical labels
class_mapping = {'DoS': 0, 'Fuzzers': 1, 'Generic': 2, 'Exploits': 3, 'Reconnaissance': 4, 'Analysis': 5, 'Shellcode': 6, 'Worms': 7, 'Backdoor': 8, 'Normal': 9}
y = y.map(class_mapping)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# For numerical columns: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Encode categorical columns with LabelEncoder
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

# Apply transformations to appropriate columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols)
    ])

# Create a pipeline with preprocessing and Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the dataset into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
