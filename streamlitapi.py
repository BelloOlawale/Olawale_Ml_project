import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Lazy-load model components ---
model = None
scaler = None
model_columns = None

# --- File paths ---
MODEL_PATH = "logistic_regression_model.pkl"
SCALER_PATH = "scaler.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"

# --- Page setup ---
st.set_page_config(page_title="Bank Term Deposit Prediction App", page_icon="🏦", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4, h5 {
        color: #4B9CD3 !important;
        font-weight: 700;
    }
    label, .stMarkdown {
        color: #EAEAEA !important;
        font-size: 16px;
    }
    div[data-testid="stNumberInput"] label,
    .stRadio label {
        color: #FFFFFF !important;
    }
    hr {
        border: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align:center;'>🏦 Bank Term Deposit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#AAAAAA;'>Predict whether a customer will subscribe to a term deposit based on their details.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Customer Info ---
st.markdown("<h2>📋 Customer Information</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    balance = st.number_input("Account Balance (€)", value=1000)
    campaign = st.number_input("Number of Contacts During Campaign", value=2)
with col2:
    duration = st.number_input("Last Contact Duration (seconds)", value=100)
    pdays = st.number_input("Days Since Last Contact (-1 means never contacted)", value=-1)
    previous = st.number_input("Number of Previous Contacts", value=0)

st.markdown("---")

# --- Campaign & Personal Details ---
st.markdown("<h2>🎯 Campaign & Personal Details</h2>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("<h5>Job</h5>", unsafe_allow_html=True)
    job = st.radio("", 
                   ["admin.", "blue-collar", "entrepreneur", "services", "technician", "management", 
                    "retired", "student", "unemployed", "housemaid", "self-employed"], 
                   key="job_radio")

    st.markdown("<h5>Marital Status</h5>", unsafe_allow_html=True)
    marital = st.radio("", ["married", "single", "divorced"], key="marital_radio")

    st.markdown("<h5>Education</h5>", unsafe_allow_html=True)
    education = st.radio("", ["primary", "secondary", "tertiary"], key="education_radio")

with col4:
    st.markdown("<h5>Housing Loan?</h5>", unsafe_allow_html=True)
    housing = st.radio("", ["yes", "no"], horizontal=True, key="housing_radio")

    st.markdown("<h5>Personal Loan?</h5>", unsafe_allow_html=True)
    loan = st.radio("", ["yes", "no"], horizontal=True, key="loan_radio")

    st.markdown("<h5>Previous Outcome</h5>", unsafe_allow_html=True)
    poutcome = st.radio("", ["unknown", "failure", "success"], horizontal=True, key="poutcome_radio")

st.markdown("---")

# --- Helper function for encoding ---
def encode_input(job, marital, education, housing, loan, poutcome):
    mappings = {
        "job": {"admin.": 0, "blue-collar": 1, "entrepreneur": 2, "services": 3, "technician": 4, "management": 5,
                "retired": 6, "student": 7, "unemployed": 8, "housemaid": 9, "self-employed": 10},
        "marital": {"married": 0, "single": 1, "divorced": 2},
        "education": {"primary": 0, "secondary": 1, "tertiary": 2},
        "housing": {"yes": 1, "no": 0},
        "loan": {"yes": 1, "no": 0},
        "poutcome": {"unknown": 0, "failure": 1, "success": 2}
    }
    return [
        mappings["job"][job],
        mappings["marital"][marital],
        mappings["education"][education],
        mappings["housing"][housing],
        mappings["loan"][loan],
        mappings["poutcome"][poutcome]
    ]

# --- Predict button ---
if st.button("🔍 Predict Subscription"):
    try:
        # Lazy-load model if not already loaded
        if model is None:
            model = joblib.load(MODEL_PATH)
        if scaler is None:
            scaler = joblib.load(SCALER_PATH)
        if model_columns is None:
            model_columns = joblib.load(MODEL_COLUMNS_PATH)
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Please train the model first (run ML.py).")
        st.stop()
    except ImportError:
        st.error("Missing dependency. Please install with `pip install scikit-learn scipy`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    # --- Prepare input data ---
    input_raw = pd.DataFrame({
        'age': [age],
        'balance': [balance],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'housing': [housing],
        'loan': [loan],
        'poutcome': [poutcome]
    })

    try:
        input_encoded = pd.get_dummies(input_raw, drop_first=True)
        input_aligned = input_encoded.reindex(columns=list(model_columns), fill_value=0)
        scaled_input = scaler.transform(input_aligned)

        # --- Make prediction ---
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction[0] == 1:
            st.success(f"✅ The customer is likely to SUBSCRIBE! (Confidence: {probability:.2%})")
        else:
            st.error(f"❌ The customer is unlikely to subscribe. (Confidence: {1 - probability:.2%})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown("""
    <hr style='margin-top: 50px;'>
    <p style='text-align:center; color:#AAAAAA; font-size:14px;'>
    Developed by <b style='color:#4B9CD3;'>Olawale Bello</b> | Data Science Intern, Wragby Business Solutions<br>
    <span style='font-size:12px;'>Built with ❤️ using Streamlit</span>
    </p>
""", unsafe_allow_html=True)


















