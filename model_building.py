import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load Dataset ---
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cultivar'] = data.target

# --- 2. Feature Selection ---
# We select 6 specific features as required
selected_features = [
    'alcohol', 
    'flavanoids', 
    'color_intensity', 
    'hue', 
    'proline', 
    'magnesium'
]

X = df[selected_features]
y = df['cultivar']

# --- 3. Preprocessing ---
# Handle missing values (fillna with mean if any exist)
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Pipeline Construction ---
# Combine Scaler and Model into one object
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 5. Train Model ---
pipeline.fit(X_train, y_train)

# --- 6. Evaluate Model ---
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 7. Save Model ---
# This ensures it saves in the same folder as this notebook
current_directory = os.getcwd()
filename = 'wine_cultivar_model.pkl'
full_path = os.path.join(current_directory, filename)

joblib.dump(pipeline, full_path)
print(f"Model successfully saved to: {full_path}")