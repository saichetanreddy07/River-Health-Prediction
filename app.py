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
# Note: Ensure these file names match the output of your final training step.
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

with st.spinner('Loading prediction model...'):
    try:
        best_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        encoder = joblib.load(ENCODER_FILE)
        model_loaded = True
    except FileNotFoundError:
        st.error(
            f"🚨 Model files not found. Please ensure '{MODEL_FILE}', '{SCALER_FILE}', and '{ENCODER_FILE}' are in the same directory.")
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
**Analyze the potential impact of industrial discharge on river ecosystems using 7 key environmental parameters.**
""")

with st.expander("🔬 Learn About This Predictor"):
    st.markdown("""
    This application uses a **Machine Learning model** (likely Random Forest or MLP) to estimate the **probability of high pollution** in a river segment. The model was trained on historical data relating **Industrial Type** and **SIX** critical water quality parameters:

    * **pH**
    * **Nitrate**
    * **Water Temperature**
    * **Turbidity**
    * **Dissolved Oxygen (DO)**
    * **Conductivity**

    This expanded feature set provides a much more robust risk assessment for monitoring efforts.
    """)

st.markdown("---")

# --- Input Area ---
st.header("1. Enter Water Quality Parameters")
st.info("Adjust the sliders and input fields below to simulate a scenario. Note the new features added!")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Industry & Acidity")
    industry_type = st.selectbox(
        "🏭 Industry Type",
        ['chemical', 'food_processing', 'textile'],
        help="Select the dominant industrial activity near the sampling site."
    )

    ph = st.slider(
        "🧪 pH Level (Acidity/Alkalinity)",
        0.0, 14.0, 7.0, 0.1,
        help="Healthy rivers are typically between 6.5 and 8.5. Extremes can indicate industrial waste."
    )

with col2:
    st.subheader("Contaminants & Heat")
    nitrate = st.number_input(
        "📉 Nitrate (mg/L)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.2f",
        help="Contaminant level. High values often indicate sewage or chemical runoff."
    )

    water_temperature = st.slider(
        "🌡️ Water Temperature (°C)",
        0.0, 40.0, 20.0, 0.5,
        help="Affects dissolved oxygen and biological processes. Industrial cooling water can raise this."
    )

with col3:
    st.subheader("Physical Health")
    turbidity = st.slider(
        "🌫️ Turbidity (NTU)",
        0.0, 150.0, 10.0, 1.0,
        help="A measure of water clarity. High turbidity can be caused by suspended solids from erosion or heavy discharge."
    )

    do = st.slider(
        "🌬️ Dissolved Oxygen (DO, mg/L)",
        0.0, 14.0, 8.0, 0.1,
        help="Critical for aquatic life. Low DO (below 5 mg/L) is often a sign of organic pollution."
    )

    conductivity = st.slider(
        "⚡ Conductivity (µS/cm)",
        0.0, 2000.0, 300.0, 10.0,
        help="Measures the water's ability to pass electric current. High conductivity often indicates high concentrations of dissolved ions/salts from industrial discharge."
    )

st.markdown("---")

# --- Prediction & Output Area ---
st.header("2. Get Prediction")

if st.button("🚀 Predict Pollution Risk", type="primary"):

    # 1. Prepare Input Data (MATCHING THE 7 MODEL FEATURES EXACTLY)
    input_data = pd.DataFrame({
        'Industry_Type': [industry_type],
        'pH': [ph],
        'Nitrate': [nitrate],  # Renamed column
        'Water_Temperature': [water_temperature],  # Renamed column
        'Turbidity': [turbidity],  # New column
        'Dissolved_Oxygen': [do],  # New column
        'Conductivity': [conductivity]  # New column
    })

    try:
        # 2. Preprocess
        # Create a copy to avoid Streamlit mutability issues
        input_processed = input_data.copy()

        # Encode Industry_Type (Must be 2D array if encoder expects it)
        try:
            input_processed['Industry_Type'] = encoder.transform(input_processed[['Industry_Type']])
        except ValueError:
            # Fallback for older versions/different encoder fits
            input_processed['Industry_Type'] = encoder.transform(input_processed['Industry_Type'])

        # Scale all features (Must match the order and number of features used for scaler fit)
        input_scaled = scaler.transform(input_processed)

        # 3. Make Prediction
        probability = None
        if hasattr(best_model, 'predict_proba'):
            # Works for Random Forest, Logistic Regression, SVM (with probability=True)
            probability = best_model.predict_proba(input_scaled)[0, 1] * 100
        elif hasattr(best_model, 'predict'):
            # Works for MLP/LSTM (if they output a single value 0-1)
            prediction_output = best_model.predict(input_scaled)
            if prediction_output.ndim == 2 and prediction_output.shape[1] == 1:
                probability = prediction_output[0, 0] * 100
            else:
                probability = prediction_output[0] * 100
        else:
            st.error("The loaded model does not support probability prediction.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check that the feature order in the input DataFrame matches the training data.")
        probability = None

    # 4. Display Results
    if probability is not None:
        st.subheader("📊 Prediction Analysis")
        st.write(f"The predicted **Probability of High Pollution** is:")

        # Define categories and colors for visualization
        if probability <= 20:
            category = "Very Low Risk"
            emoji = "✅"
            color = "#28a745"  # Bootstrap success green
            message = "This scenario suggests the river is likely healthy. Maintain vigilance!"
        elif probability <= 40:
            category = "Low Risk"
            emoji = "👍"
            color = "#17a2b8"  # Bootstrap info blue
            message = "Water quality appears good, but minor factors may warrant basic monitoring."
        elif probability <= 60:
            category = "Moderate Risk"
            emoji = "⚠️"
            color = "#ffc107"  # Bootstrap warning yellow
            message = "Pollution factors are present. Enhanced monitoring and potential mitigation steps are recommended."
        elif probability <= 80:
            category = "High Risk"
            emoji = "🔥"
            color = "#dc3545"  # Bootstrap danger red
            message = "Significant pollution risk detected. Immediate investigation and corrective action are required."
        else:
            category = "Very High Risk"
            emoji = "🛑"
            color = "#6f42c1"  # Bootstrap dark purple/critical
            message = "🚨 Critical risk! Pollution is highly probable. Stop discharge and initiate cleanup protocols."

        # Use markdown with color for emphasis
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: {'black' if color in ['#ffc107', '#17a2b8'] else 'white'};
            box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);
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
st.caption(
    "Disclaimer: This tool provides an estimate based on a predictive model and should not replace professional environmental analysis or mandatory testing.")
