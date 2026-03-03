import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🔮",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Animated starfield background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0d1b2a 0%, #050a0f 100%);
    min-height: 100vh;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 70%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 15%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 55%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 90%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 30%, rgba(0,245,255,0.6) 0%, transparent 100%),
        radial-gradient(2px 2px at 20% 45%, rgba(0,255,148,0.5) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

/* Hero Banner */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,245,255,0.1);
    border: 1px solid rgba(0,245,255,0.3);
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #00f5ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2rem, 5vw, 3.8rem);
    font-weight: 900;
    background: linear-gradient(135deg, #00f5ff 0%, #00ff94 50%, #7b2fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.45);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 1px;
}

/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f5ff, #00ff94, transparent);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* Glass Cards */
.glass-card {
    background: rgba(255,255,255,0.03);
    border-radius: 24px;
    padding: 28px 28px 10px;
    border: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
    transition: box-shadow 0.3s ease, border 0.3s ease;
}
.glass-card:hover {
    box-shadow: 0 0 40px rgba(0,245,255,0.12), 0 8px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(0,245,255,0.2);
}
.card-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    color: #00f5ff;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,245,255,0.3), transparent);
}

/* Streamlit widget label overrides */
[data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }
div[data-baseweb="slider"] > div > div > div { background: linear-gradient(90deg, #00f5ff, #00ff94) !important; }
div[data-baseweb="slider"] [role="slider"] {
    background: #00f5ff !important;
    box-shadow: 0 0 12px #00f5ff !important;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    height: 3.5em;
    border-radius: 14px;
    background: linear-gradient(135deg, #00f5ff 0%, #00ff94 50%, #7b2fff 100%) !important;
    color: #050a0f !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 0 40px rgba(0,245,255,0.6), 0 0 80px rgba(0,255,148,0.3) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* Result Card */
.result-safe {
    background: linear-gradient(135deg, rgba(0,255,148,0.08), rgba(0,245,255,0.05));
    border: 1px solid rgba(0,255,148,0.3);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(0,255,148,0.1);
}
.result-risk {
    background: linear-gradient(135deg, rgba(255,50,80,0.1), rgba(255,100,50,0.06));
    border: 1px solid rgba(255,50,80,0.35);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(255,50,80,0.1);
}
.result-icon { font-size: 4rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-label-safe { color: #00ff94; }
.result-label-risk { color: #ff3250; }
.result-prob {
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -1px;
}
.result-prob-safe { color: #00ff94; }
.result-prob-risk { color: #ff3250; }
.result-desc {
    color: rgba(255,255,255,0.5);
    font-size: 0.9rem;
    margin-top: 0.75rem;
    max-width: 340px;
    margin-left: auto;
    margin-right: auto;
}

/* Progress bar override */
[data-testid="stProgress"] > div > div > div {
    border-radius: 20px;
    background: linear-gradient(90deg, #00f5ff, #00ff94) !important;
}
[data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 20px;
}

/* Metric strip */
.metric-strip {
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.metric-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 10px 20px;
    text-align: center;
    min-width: 130px;
}
.metric-pill-label { font-size: 0.7rem; color: rgba(255,255,255,0.4); letter-spacing: 1px; text-transform: uppercase; }
.metric-pill-value { font-size: 1.1rem; font-weight: 700; color: #fff; margin-top: 2px; }

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model2.h5')
    with open('sc.pkl', 'rb') as f: sc = pickle.load(f)
    with open('label.pkl', 'rb') as f: le = pickle.load(f)
    with open('ohe.pkl', 'rb') as f: ohe = pickle.load(f)
    return model, sc, le, ohe

model, sc, le, ohe = load_artifacts()

# ---------------- HERO ----------------
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🔮 AI-Powered Analytics</div>
    <div class='hero-title'>CHURN INTELLIGENCE</div>
    <div class='hero-subtitle'>Predict customer retention with deep neural precision</div>
</div>
<div class='neon-divider'></div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='glass-card'><div class='card-header'>👤 Customer Profile</div>", unsafe_allow_html=True)
    geography     = st.selectbox('Geography', ohe.categories_[0])
    gender        = st.selectbox('Gender', le.classes_)
    age           = st.slider('Age', 18, 92, 35)
    tenure        = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Products Held', 1, 4, 1)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'><div class='card-header'>💳 Financial Profile</div>", unsafe_allow_html=True)
    credit_score      = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    balance           = st.number_input('Account Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary  = st.number_input('Estimated Salary ($)', min_value=0.0, value=60000.0, step=1000.0)
    has_cr_card       = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "✅ Yes" if x else "❌ No")
    is_active_member  = st.selectbox('Active Member', [0, 1], format_func=lambda x: "✅ Yes" if x else "❌ No")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICT ----------------
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("⚡ ANALYSE & PREDICT")

# ---------------- RESULT ----------------
if predict:
    with st.spinner("Running neural inference..."):
        time.sleep(0.8)  # brief dramatic pause

    input_df = pd.DataFrame({
        'CreditScore': [credit_score], 'Geography': [geography],
        'Gender': [gender], 'Age': [age], 'Tenure': [tenure],
        'Balance': [balance], 'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card], 'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    input_df['Gender'] = le.transform(input_df['Gender'])
    geo_encoded = ohe.transform(input_df[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)
    input_scaled = sc.transform(input_df)

    prob = float(model.predict(input_scaled)[0][0])
    churned = prob > 0.5

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    _, res_col, _ = st.columns([1, 3, 1])
    with res_col:
        if churned:
            st.markdown(f"""
            <div class='result-risk'>
                <div class='result-icon'>🚨</div>
                <div class='result-label result-label-risk'>HIGH CHURN RISK</div>
                <div class='result-prob result-prob-risk'>{prob:.1%}</div>
                <div class='result-desc'>
                    This customer shows strong indicators of leaving.
                    Immediate retention action is recommended.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
                <div class='result-icon'>🛡️</div>
                <div class='result-label result-label-safe'>LOW CHURN RISK</div>
                <div class='result-prob result-prob-safe'>{prob:.1%}</div>
                <div class='result-desc'>
                    This customer appears stable and satisfied.
                    Continue engagement to maintain loyalty.
                </div>
            </div>""", unsafe_allow_html=True)

        # Confidence progress bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("MODEL CONFIDENCE")
        st.progress(prob if churned else 1 - prob)

        # Metric pills
        risk_tier = "CRITICAL" if prob > 0.75 else ("HIGH" if prob > 0.5 else ("MODERATE" if prob > 0.35 else "LOW"))
        st.markdown(f"""
        <div class='metric-strip'>
            <div class='metric-pill'>
                <div class='metric-pill-label'>Churn Prob.</div>
                <div class='metric-pill-value'>{prob:.2%}</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-pill-label'>Risk Tier</div>
                <div class='metric-pill-value'>{risk_tier}</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-pill-label'>Retention Prob.</div>
                <div class='metric-pill-value'>{1-prob:.2%}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>CHURN INTELLIGENCE · Powered by Deep Learning · © 2026</div>", unsafe_allow_html=True)