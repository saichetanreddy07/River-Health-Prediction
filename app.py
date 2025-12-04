import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="River Pollution Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session storage for past predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------
# Load the trained model
# ----------------------------
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

with st.spinner('Loading the river pollution model…'):
    try:
        best_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        encoder = joblib.load(ENCODER_FILE)
        model_ready = True
    except FileNotFoundError:
        st.error(
            "⚠️ Model files missing. Please make sure 'best_model.pkl', 'scaler.pkl', and 'encoder.pkl' are in the same folder as this app."
        )
        model_ready = False
    except Exception as e:
        st.error(f"Something went wrong while loading the model: {e}")
        model_ready = False

if not model_ready:
    st.stop()

# ----------------------------
# Introduction
# ----------------------------
st.title("💧 River Pollution Risk Predictor")
st.markdown("""
Evaluate how industrial activity and water quality might affect a river’s health—using real environmental science and machine learning.
""")

with st.expander("🔍 How This Tool Works"):
    st.markdown("""
    This app uses a trained machine learning model to estimate the **likelihood of high pollution** based on:

    - The **type of nearby industry** (chemical, food processing, or textile)
    - **Six key water quality measurements**:
      - pH (acidity/alkalinity)
      - Nitrate levels
      - Water temperature
      - Turbidity (cloudiness)
      - Dissolved oxygen (essential for fish and insects)
      - Conductivity (a sign of dissolved salts or pollutants)

    The model was built from historical water monitoring data and gives a percentage risk score—helping you prioritize which sites need closer inspection.
    """)

st.markdown("---")

# ----------------------------
# User Inputs
# ----------------------------
st.header("1. Describe the River Site")

st.info("Use the controls below to reflect real or hypothetical conditions at your monitoring location.")

# Split into two columns for better readability
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏭 Nearby Industry & Chemistry")
    industry_type = st.selectbox(
        "What kind of industry is nearby?",
        options=['chemical', 'food_processing', 'textile'],
        help="Choose the dominant industrial activity affecting the river."
    )

    ph = st.slider(
        "pH Level (0 = very acidic, 14 = very alkaline)",
        min_value=0.0, max_value=14.0, value=7.0, step=0.1,
        help="Natural rivers usually range from 6.5 to 8.5."
    )

    nitrate = st.number_input(
        "Nitrate (mg/L)",
        min_value=0.0, max_value=100.0, value=5.0, step=0.1,
        format="%.2f",
        help="High nitrate often comes from fertilizers, sewage, or industrial waste."
    )

    water_temperature = st.slider(
        "Water Temperature (°C)",
        min_value=0.0, max_value=40.0, value=20.0, step=0.5,
        help="Warmer water holds less oxygen and may indicate thermal pollution."
    )

with col_b:
    st.subheader("💧 Physical Water Conditions")
    turbidity = st.slider(
        "Turbidity (NTU – cloudiness)",
        min_value=0.0, max_value=150.0, value=10.0, step=1.0,
        help="High turbidity blocks sunlight and harms aquatic plants."
    )

    do = st.slider(
        "Dissolved Oxygen (mg/L)",
        min_value=0.0, max_value=14.0, value=8.0, step=0.1,
        help="Healthy rivers typically have >6 mg/L. Below 5 mg/L stresses aquatic life."
    )

    conductivity = st.slider(
        "Conductivity (µS/cm)",
        min_value=0.0, max_value=2000.0, value=300.0, step=10.0,
        help="High values suggest dissolved ions—often from industrial discharge or road salt."
    )

st.markdown("---")

# ----------------------------
# Prediction
# ----------------------------
st.header("2. Get Your Risk Assessment")

if st.button("🚀 Analyze Pollution Risk", type="primary"):
    # Prepare input
    input_df = pd.DataFrame({
        'Industry_Type': [industry_type],
        'pH': [ph],
        'Nitrate': [nitrate],
        'Water_Temperature': [water_temperature],
        'Turbidity': [turbidity],
        'Dissolved_Oxygen': [do],
        'Conductivity': [conductivity]
    })

    try:
        # Encode and scale
        input_copy = input_df.copy()
        encoded_industry = encoder.transform(input_copy[['Industry_Type']])
        input_copy['Industry_Type'] = encoded_industry
        scaled_input = scaler.transform(input_copy)

        # Predict
        if hasattr(best_model, 'predict_proba'):
            risk_prob = best_model.predict_proba(scaled_input)[0, 1] * 100
        else:
            pred = best_model.predict(scaled_input)
            risk_prob = (pred[0] if pred.ndim == 1 else pred[0, 0]) * 100

    except Exception as e:
        st.error(f"Oops—something went wrong during prediction: {e}")
        risk_prob = None

    # Save to history
    if risk_prob is not None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.prediction_history.append({
            "Time": now,
            "Industry": industry_type,
            "pH": ph,
            "Nitrate (mg/L)": nitrate,
            "Temp (°C)": water_temperature,
            "Turbidity (NTU)": turbidity,
            "DO (mg/L)": do,
            "Conductivity (µS/cm)": conductivity,
            "Pollution Risk (%)": round(risk_prob, 2)
        })

        # Display result
        st.subheader("📊 Your Pollution Risk Assessment")

        if risk_prob <= 20:
            label, icon, color, advice = "Very Low Risk", "✅", "#28a745", "The river appears healthy. Keep monitoring to stay ahead of changes."
        elif risk_prob <= 40:
            label, icon, color, advice = "Low Risk", "👍", "#17a2b8", "Water quality is generally good, but stay alert for small changes."
        elif risk_prob <= 60:
            label, icon, color, advice = "Moderate Risk", "⚠️", "#ffc107", "Some warning signs are present. Consider follow-up testing or site review."
        elif risk_prob <= 80:
            label, icon, color, advice = "High Risk", "🔥", "#dc3545", "Strong indicators of pollution. Investigate sources and take action soon."
        else:
            label, icon, color, advice = "Very High Risk", "🛑", "#6f42c1", "Critical alert: pollution is highly likely. Immediate response needed."

        # Highlight result
        text_color = "black" if color in ["#ffc107", "#17a2b8"] else "white"
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: {text_color};
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            <h3>{icon} {label} {icon}</h3>
            <h1>{risk_prob:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.write(advice)

        if "Low" in label:
            st.success("✅ No immediate concerns—great work protecting this river!")
        else:
            st.error(
                "❗ This situation warrants attention. Consider notifying environmental authorities or scheduling a field test.")

st.markdown("---")

# ----------------------------
# Historical Dashboard
# ----------------------------
st.header("3. 📊 Your Prediction History")

if st.session_state.prediction_history:
    history = pd.DataFrame(st.session_state.prediction_history)
    history = history.sort_values("Time", ascending=False).reset_index(drop=True)

    st.dataframe(history, use_container_width=True, hide_index=True)

    if len(history) > 1:
        st.subheader("Risk Over Time")
        st.line_chart(history.set_index("Time")["Pollution Risk (%)"])
else:
    st.info("You haven’t run any predictions yet. Try entering some values above to get started!")

st.caption(
    "ℹ️ Note: This tool provides an estimated risk based on a predictive model. It is not a substitute for certified water quality testing or regulatory compliance checks."
)
