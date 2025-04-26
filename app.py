import streamlit as st
import joblib
import pandas as pd

# Load saved model, scaler, and encoder
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="River Pollution Predictor", layout="centered")
st.title("🌊 River Pollution Prediction App")

# Input fields
factory_industry = st.selectbox("Select Factory Industry", encoder.classes_)
ph = st.number_input("Enter pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
nitrate = st.number_input("Enter Nitrate Level", min_value=0.0, value=10.0, step=0.1)
temperature = st.number_input("Enter Water Temperature (°C)", min_value=0.0, value=25.0, step=0.1)

# Predict button
if st.button("Predict Pollution Level"):
    try:
        # Encode and scale
        encoded_industry = encoder.transform([factory_industry])[0]
        input_data = pd.DataFrame([[encoded_industry, ph, nitrate, temperature]],
                                  columns=["factory_industry", "ph", "nitrate", "temperature"])
        input_scaled = scaler.transform(input_data)

        # Predict
        probability = model.predict_proba(input_scaled)[0, 1] * 100

        # Category
        if probability <= 20:
            category = "🟢 Very Low Pollution"
        elif probability <= 40:
            category = "🟡 Low Pollution"
        elif probability <= 60:
            category = "🟠 Moderate Pollution"
        elif probability <= 80:
            category = "🔴 High Pollution"
        else:
            category = "🔴🔴 Very High Pollution"

        st.success(f"Predicted Pollution Probability: **{probability:.2f}%**")
        st.markdown(f"### Category: {category}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
