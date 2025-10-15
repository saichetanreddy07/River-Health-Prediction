import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration for a better initial look
st.set_page_config(
    page_title="River Pollution Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Model Loading ---
with st.spinner('Loading prediction model...'):
    try:
        best_model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("encoder.pkl")
        model_loaded = True
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please ensure 'best_model.pkl', 'scaler.pkl', and 'encoder.pkl' are in the same directory.")
        model_loaded = False
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        model_loaded = False

if not model_loaded:
    st.stop()


# ====================================
# --- UI/UX & BACKGROUND ENHANCEMENTS ---
# ====================================

# --- Header & Background Section ---
st.title("💧 River Water Quality and Pollution Predictor")
st.markdown("""
**Analyze the potential impact of industrial discharge on river ecosystems.**
""")

with st.expander("🔬 Learn About This Predictor"):
    st.markdown("""
    This application uses a **Machine Learning model** to estimate the **probability of high pollution** in a river segment, based on key water quality parameters and the type of nearby industrial activity.

    ### Why is this important?
    * **Environmental Protection:** Identifying potential pollution sources is crucial for maintaining aquatic biodiversity and ecosystem health.
    * **Regulatory Compliance:** Helps industries and regulators assess risk and enforce water quality standards.
    * **Data-Driven Decisions:** Provides a quick, quantitative risk assessment for monitoring efforts.

    ### How It Works:
    The model was trained on historical data relating **Industrial Type**, **pH**, **Nitrate Concentration**, and **Temperature** to observed river pollution levels. It outputs a percentage probability, which is then mapped to a pollution risk category.
    """)

st.markdown("---")

# --- Input Area ---
st.header("1. Enter Water Quality Parameters")
st.info("Adjust the sliders and input fields below to simulate a scenario.")

col1, col2 = st.columns(2)

with col1:
    industry_type = st.selectbox(
        "🏭 Industry Type",
        ['chemical', 'food_processing', 'textile'],
        help="Select the dominant industrial activity near the sampling site."
    )

    ph = st.slider(
        "🧪 pH Level (Acidity/Alkalinity)",
        0.0, 14.0, 7.0, 0.1,
        help="A measure of water's acidity. Healthy rivers are typically between 6.5 and 8.5."
    )

with col2:
    nitrate = st.number_input(
        "📉 Nitrate Concentration (mg/L)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.2f",
        help="High levels can indicate runoff from agriculture or sewage/industrial waste."
    )

    temperature = st.slider(
        "🌡️ Water Temperature (°C)",
        0.0, 40.0, 20.0, 0.5,
        help="Affects dissolved oxygen levels and biological activity. High temps can stress aquatic life."
    )

st.markdown("---")

# --- Prediction & Output Area ---
st.header("2. Get Prediction")

if st.button("🚀 Predict Pollution Risk", type="primary"):
    # 1. Prepare Input Data
    input_data = pd.DataFrame({
        'Industry_Type': [industry_type],
        'pH': [ph],
        'Nitrate_Concentration': [nitrate],
        'Temperature': [temperature]
    })

    try:
        # 2. Preprocess
        input_data['Industry_Type'] = encoder.transform(input_data[['Industry_Type']])
        input_scaled = scaler.transform(input_data)

        # 3. Make Prediction
        probability = None
        if hasattr(best_model, 'predict_proba'):
            probability = best_model.predict_proba(input_scaled)[0, 1] * 100
        elif hasattr(best_model, 'predict'):
            prediction_output = best_model.predict(input_scaled)
            if isinstance(prediction_output, np.ndarray) and prediction_output.ndim == 2:
                 probability = prediction_output[0, 0] * 100
            else:
                 probability = prediction_output[0] * 100
        else:
            st.error("The loaded model does not support probability prediction.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        probability = None


    # 4. Display Results
    if probability is not None:
        st.subheader("📊 Prediction Analysis")
        st.write(f"The predicted **Probability of High Pollution** is:")

        # Define categories and colors for visualization
        if probability <= 20:
            category = "Very Low Risk"
            emoji = "✅"
            color = "green"
            message = "This scenario suggests the river is likely healthy. Maintain vigilance!"
        elif probability <= 40:
            category = "Low Risk"
            emoji = "👍"
            color = "lightgreen"
            message = "Water quality appears good, but minor factors may warrant basic monitoring."
        elif probability <= 60:
            category = "Moderate Risk"
            emoji = "⚠️"
            color = "orange"
            message = "Pollution factors are present. Enhanced monitoring and potential mitigation steps are recommended."
        elif probability <= 80:
            category = "High Risk"
            emoji = "🔥"
            color = "red"
            message = "Significant pollution risk detected. Immediate investigation and corrective action are required."
        else:
            category = "Very High Risk"
            emoji = "🛑"
            color = "darkred"
            message = "🚨 Critical risk! Pollution is highly probable. Stop discharge and initiate cleanup protocols."

        # Use markdown with color for emphasis
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: {'black' if color in ['lightgreen', 'green', 'orange'] else 'white'};
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        ">
            <h3>{emoji} {category} {emoji}</h3>
            <h1>{probability:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### Pollution Category: **{category}**")
        st.write(message)

        # Final action-oriented message
        if 'Low' in category or 'Very Low' in category:
            st.success("The river segment appears to be in good health based on the inputs.")
        else:
            st.error("Action may be required! This scenario indicates a potential environmental threat.")

st.markdown("---")
st.caption("Disclaimer: This tool provides an estimate based on a predictive model and should not replace professional environmental analysis or mandatory testing.")
