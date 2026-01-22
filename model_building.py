import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cultivar'] = data.target

# 2. Feature Selection (Selecting 6 features)
# We select these because they often have high importance in distinguishing wines
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

# 3. Data Preprocessing
# Check for missing values (Wine dataset usually has none, but good practice)
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline Construction (Scaling + Modeling)
# We use a Pipeline so the scaler is saved WITH the model.
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Mandatory scaling
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train the Model
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the Model
# We save the whole pipeline (Scaler + Model)
joblib.dump(pipeline, 'wine_cultivar_model.pkl')
print("Model saved as wine_cultivar_model.pkl")
