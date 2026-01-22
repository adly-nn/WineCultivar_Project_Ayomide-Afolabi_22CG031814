import streamlit as st
import joblib
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷")

# --- Load Model Function ---
@st.cache_resource
def load_model():
    # Construct path relative to app.py
    # app.py is in root, model is in /model/wine_cultivar_model.pkl
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'wine_cultivar_model.pkl')
    return joblib.load(model_path)

# --- App Logic ---
try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run 'model_building.ipynb' first to generate the model.")
    st.stop()

st.title("🍷 Wine Cultivar Origin Predictor")
st.markdown("Enter the chemical properties below to predict the wine's origin.")

# --- Input Form ---
with st.form("wine_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        alcohol = st.number_input("Alcohol", min_value=11.0, max_value=15.0, value=13.0, step=0.1)
        flavanoids = st.number_input("Flavanoids", min_value=0.3, max_value=5.1, value=2.0, step=0.1)
        color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=4.5, step=0.1)
        
    with col2:
        hue = st.number_input("Hue", min_value=0.4, max_value=1.8, value=1.0, step=0.01)
        proline = st.number_input("Proline", min_value=270.0, max_value=1700.0, value=750.0, step=10.0)
        magnesium = st.number_input("Magnesium", min_value=70.0, max_value=170.0, value=100.0, step=1.0)
    
    submitted = st.form_submit_button("Predict Cultivar")

# --- Prediction ---
if submitted:
    try:
        # Create array in exact order of training features
        features = np.array([alcohol, flavanoids, color_intensity, hue, proline, magnesium]).reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        
        # Map output to friendly names (0,1,2 -> Cultivar 1,2,3)
        cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
        result = cultivar_map.get(prediction, "Unknown")
        
        st.success(f"✅ Predicted Origin: **{result}**")
        
    except Exception as e:
        st.error(f"Error: {e}")

st.sidebar.info("System: Random Forest Classifier | Dataset: UCI Wine")