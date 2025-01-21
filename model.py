import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess the data
heart_data = pd.read_csv("D:/5th semester/Machine Learning EE5252/Heart_Disease_Flask_app/heart.csv")

x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Save the model for later use
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)
