import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing objects
try:
    best_model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'best_model.pkl', 'scaler.pkl', and 'encoder.pkl' are in the same directory.")
    st.stop()


st.title("River Pollution Prediction")

st.write("Enter the water quality parameters to predict the pollution level.")

# Input fields for user to enter data
industry_type = st.selectbox("Industry Type", ['food_processing', 'chemical', 'textile'])
ph = st.slider("pH Level", 0.0, 14.0, 7.0)
nitrate = st.number_input("Nitrate Concentration (mg/L)", min_value=0.0)
temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0)

# Prediction button
if st.button("Predict Pollution"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Industry_Type': [industry_type],
        'pH': [ph],
        'Nitrate_Concentration': [nitrate],
        'Temperature': [temperature]
    })

    # Encode the industry type
    input_data['Industry_Type'] = encoder.transform(input_data['Industry_Type'])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(input_scaled)[0, 1] * 100
    elif hasattr(best_model, 'predict'):
        # For Keras models, predict returns probabilities directly for binary classification
        probability = best_model.predict(input_scaled)[0, 0] * 100
    else:
        st.error("The loaded model does not support probability prediction.")
        probability = None

    if probability is not None:
        st.subheader("Prediction Result:")
        if probability <= 20:
            category = "Very Low Pollution"
        elif probability <= 40:
            category = "Low Pollution"
        elif probability <= 60:
            category = "Moderate Pollution"
        elif probability <= 80:
            category = "High Pollution"
        else:
            category = "Very High Pollution"

        st.write(f"Predicted Probability of Pollution: **{probability:.2f}%**")
        st.write(f"Pollution Category: **{category}**")

        # Display a simple indicator based on the category
        if 'Low' in category or 'Very Low' in category:
            st.success("River health is likely good.")
        else:
            st.warning("Potential pollution detected.")