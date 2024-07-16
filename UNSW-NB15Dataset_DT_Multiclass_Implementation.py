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
import lime
from lime import lime_text
from lime.lime_tabular import LimeTabularExplainer
np.random.seed(1)
import webbrowser

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

# Split the dataset into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Preprocessing done")

# Define and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train_processed, y_train)
print("training done")

print("LIMEing")
#LIME for explanations
# Get feature names after preprocessing
feature_names=numerical_cols.tolist() + categorical_cols.tolist()

# Initialize LIME explainer
explainer = LimeTabularExplainer(X_train_processed, mode='classification',
                                 feature_names=feature_names, class_names=np.unique(y_train),
                                 discretize_continuous=True)

# Choose an instance to explain (e.g., the first instance in the test set)
instance = X_test_processed[0]
i = np.random.randint(0, X_test_processed.shape[0])

# Explain the model's prediction for the chosen instance
explanation = explainer.explain_instance(instance, model.predict_proba, num_features=43)
#explanation = explainer.explain_instance(X_test_processed[i], model.predict_proba, num_features=20, top_labels=2) #uncomment for graph plot
explanation.show_in_notebook(show_table=True, show_all=True)
#explanation.save_to_file(r'D:/aRA/9XAIforDF/Dataset2/OneDrive20240213/CSVFiles/TrainingandTestingSets/temp.html') #uncomment for graph plot
#webbrowser.open_new_tab(r'D:/aRA/9XAIforDF/Dataset2/OneDrive20240213/CSVFiles/TrainingandTestingSets/temp.html') #uncomment for graph plot

# Print the explanation
from IPython.display import display
display(explanation.show_in_notebook(show_table=True, show_all=False))

# Get the explanation as a list of tuples
explanation_list = explanation.as_list()

# Print the explanation
for feature, weight in explanation_list:
    print(f"{feature}: {weight}")

# Extract feature names and weights from the explanation
features = [x[0] for x in explanation.as_list()]
weights = [x[1] for x in explanation.as_list()]

# Plot the feature importances
plt.figure(figsize=(20, 12))
plt.barh(features, weights, color='red')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('LIME Explanation for a Random Instance in Memory Dataset - Local Explanation')
plt.show()

print("Making predictions")
y_pred = model.predict(X_test_processed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
