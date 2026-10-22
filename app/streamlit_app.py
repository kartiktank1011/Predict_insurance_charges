"""
app/streamlit_app.py
--------------------
Streamlit web application for predicting medical insurance charges.
Loads the saved model bundle from models/best_model.pkl and presents
an interactive UI for entering patient details.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Custom CSS for premium dark UI
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0d1321 100%);
        min-height: 100vh;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(124,131,253,0.15) 0%, rgba(78,205,196,0.1) 100%);
        border: 1px solid rgba(124,131,253,0.3);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c83fd, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #a0a0b0;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(78,205,196,0.12) 0%, rgba(124,131,253,0.12) 100%);
        border: 1px solid rgba(78,205,196,0.4);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-amount {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4ecdc4, #7c83fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin: 0.4rem 0;
    }
    .result-label {
        color: #a0a0b0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    .metric-chip {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 50px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        color: #c0c0d0;
    }
    .section-header {
        color: #7c83fd;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
        margin-top: 1.5rem;
    }
    div[data-testid="stSlider"] label {
        color: #c0c0d0 !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #c0c0d0 !important;
    }
    div[data-testid="stNumberInput"] label {
        color: #c0c0d0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd 0%, #4ecdc4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        width: 100% !important;
        margin-top: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(124,131,253,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124,131,253,0.5) !important;
    }
    .info-box {
        background: rgba(124,131,253,0.08);
        border-left: 3px solid #7c83fd;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        color: #b0b0c0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ #
# Load model
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model(model_path: str = "models/best_model.pkl"):
    """Load the saved model bundle with caching."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# ------------------------------------------------------------------ #
# Prediction helper
# ------------------------------------------------------------------ #
def predict_charges(bundle: dict, user_input: dict) -> float:
    """Transform input and return predicted charge in USD."""
    df = pd.DataFrame([user_input])
    X = bundle["preprocessor"].transform(df)
    raw = bundle["model"].predict(X)[0]
    if bundle.get("log_transform_target", False):
        return float(np.expm1(raw))
    return float(raw)


# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #
def main():
    # Hero header
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏥 Insurance Charge Predictor</div>
            <div class="hero-subtitle">
                ML-powered estimation of annual medical insurance costs
                based on personal health and demographic factors.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bundle = load_model()

    if bundle is None:
        st.error(
            "⚠️ Model not found at `models/best_model.pkl`. "
            "Run `python src/tune_evaluate.py` first to train and save the model."
        )
        st.stop()

    model_name = bundle.get("model_name", "Model")
    metrics = bundle.get("test_metrics", {})

    # Model info chips
    r2_val = metrics.get("R²", 0)
    mae_val = metrics.get("MAE", 0)
    rmse_val = metrics.get("RMSE", 0)

    st.markdown(
        f"""
        <div class="info-box">
            <strong>Model:</strong> {model_name} &nbsp;|&nbsp;
            <strong>R²:</strong> {r2_val:.3f} &nbsp;|&nbsp;
            <strong>MAE:</strong> ${mae_val:,.0f} &nbsp;|&nbsp;
            <strong>RMSE:</strong> ${rmse_val:,.0f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Input form ------------------------------------------------
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=64, value=30, step=1, key="age")
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=55.0,
            value=25.0,
            step=0.1,
            format="%.1f",
            key="bmi",
        )

    with col2:
        children = st.slider(
            "Number of Children / Dependents",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            key="children",
        )
        sex = st.selectbox("Biological Sex", options=["male", "female"], key="sex")

    st.markdown('<div class="section-header">🏥 Health & Location</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        smoker = st.selectbox(
            "Smoker?",
            options=["no", "yes"],
            key="smoker",
            help="Smoking status has the largest single impact on insurance charges.",
        )

    with col4:
        region = st.selectbox(
            "US Region",
            options=["northeast", "northwest", "southeast", "southwest"],
            key="region",
        )

    # BMI indicator
    if bmi < 18.5:
        bmi_label = "Underweight 🔵"
    elif bmi < 25:
        bmi_label = "Normal ✅"
    elif bmi < 30:
        bmi_label = "Overweight 🟡"
    else:
        bmi_label = "Obese 🔴"

    st.markdown(
        f"""
        <div class="info-box" style="font-size:0.85rem;">
            BMI Classification: <strong>{bmi_label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Predict button -------------------------------------------
    if st.button("🔮 Predict Insurance Charge", key="predict_btn"):
        user_input = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }

        with st.spinner("Calculating …"):
            charge = predict_charges(bundle, user_input)

        # Determine risk tier
        if charge < 5000:
            risk, risk_color = "Low Risk", "#4ecdc4"
        elif charge < 15000:
            risk, risk_color = "Moderate Risk", "#ffd93d"
        else:
            risk, risk_color = "High Risk", "#ff6b6b"

        # Key drivers
        drivers = []
        if smoker == "yes":
            drivers.append("🚬 Smoker (+significant)")
        if bmi >= 30:
            drivers.append("⚖️ High BMI")
        if age >= 50:
            drivers.append("📅 Age ≥ 50")
        if children >= 3:
            drivers.append("👨‍👩‍👧‍👦 Large family")

        drivers_html = "  &nbsp;|&nbsp;  ".join(drivers) if drivers else "No major risk factors detected"

        st.markdown(
            f"""
            <div class="result-card">
                <span class="result-label">Estimated Annual Insurance Charge</span>
                <span class="result-amount">${charge:,.2f}</span>
                <span style="color:{risk_color}; font-weight:600; font-size:1.1rem;">{risk}</span>
                <div style="margin-top:1rem; color:#9090a8; font-size:0.88rem;">
                    Key drivers: {drivers_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show monthly equivalent
        monthly = charge / 12
        st.markdown(
            f"""
            <div style="text-align:center; color:#7070a0; margin-top:0.8rem; font-size:0.9rem;">
                ≈ <strong style="color:#c0c0d0;">${monthly:,.2f}</strong> per month
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Footer ---------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#5a5a7a; font-size:0.8rem;">
            Model trained on the <em>Medical Cost Personal Dataset</em> (Kaggle, 1,338 records).
            Predictions are estimates only and do not constitute financial or medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
