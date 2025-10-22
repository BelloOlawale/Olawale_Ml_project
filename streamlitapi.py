import streamlit as st
import joblib
import numpy as np

# Load model
# NOTE: The model file 'logistic_regression_model.pkl' must be present in the execution environment.
try:
    model = joblib.load("logistic_regression_model.pkl")
except FileNotFoundError:
    st.error("Error: The model file 'logistic_regression_model.pkl' was not found.")
    st.stop()

# Page setup
st.set_page_config(page_title="Bank Term Deposit Prediction App", page_icon="🏦", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    /* --- MODIFIED: ALL HEADERS ARE NOW BRIGHT BLUE (#4B9CD3) FOR VISIBILITY --- */
    h1, h2, h3, h4, h5 {
        color: #4B9CD3 !important; /* Changed from #FFFFFF to the theme blue */
        font-weight: 700;
    }
    label, .stMarkdown {
        color: #EAEAEA !important;
        font-size: 16px;
    }
    div[data-testid="stNumberInput"] label {
        color: #FFFFFF !important;
    }
    .stRadio label {
        color: #FFFFFF !important;
    }
    hr {
        border: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

# Title (Keeping the existing inline style which matches the new global header color)
st.markdown("<h1 style='text-align:center; color:#4B9CD3;'>🏦 Bank Term Deposit Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#AAAAAA;'>Predict whether a customer will subscribe to a term deposit based on their details.</p>", unsafe_allow_html=True)
st.markdown("---")

# CUSTOMER INFO
# This H2 now inherits the brighter color from the global CSS
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

# CAMPAIGN DETAILS
# This H2 now inherits the brighter color from the global CSS
st.markdown("<h2>🎯 Campaign & Personal Details</h2>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Job</h5>", unsafe_allow_html=True)
    job = st.radio("", 
                   ["admin.", "blue-collar", "entrepreneur", "services", "technician", "management", 
                    "retired", "student", "unemployed", "housemaid", "self-employed"], 
                   key="job_radio")

    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Marital Status</h5>", unsafe_allow_html=True)
    marital = st.radio("", ["married", "single", "divorced"], key="marital_radio")

    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Education</h5>", unsafe_allow_html=True)
    education = st.radio("", ["primary", "secondary", "tertiary"], key="education_radio")

with col4:
    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Housing Loan?</h5>", unsafe_allow_html=True)
    housing = st.radio("", ["yes", "no"], horizontal=True, key="housing_radio")

    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Personal Loan?</h5>", unsafe_allow_html=True)
    loan = st.radio("", ["yes", "no"], horizontal=True, key="loan_radio")

    # --- MODIFIED: Removed inline style to inherit the bright blue from global CSS ---
    st.markdown("<h5>Previous Outcome</h5>", unsafe_allow_html=True)
    poutcome = st.radio("", ["unknown", "failure", "success"], horizontal=True, key="poutcome_radio")

st.markdown("---")

# Encode categorical data
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

# Prepare input data
input_data = [age, balance, duration, pdays, campaign, previous] + encode_input(job, marital, education, housing, loan, poutcome)
input_array = np.array(input_data).reshape(1, -1)

# Prediction button
if st.button("🔍 Predict Subscription"):
    # Mocking prediction since the model isn't truly available in this environment
    try:
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1]
    except AttributeError:
        # Fallback for demonstration if model is not loaded correctly
        st.warning("Model file not loaded. Showing a mock prediction.")
        if age > 45 or duration > 200:
             prediction = np.array([1])
             probability = 0.85
        else:
             prediction = np.array([0])
             probability = 0.15

    if prediction[0] == 1:
        st.success(f"✅ The customer is likely to SUBSCRIBE! (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ The customer is unlikely to subscribe. (Confidence: {1 - probability:.2%})")

# Footer
st.markdown("""
    <hr style='margin-top: 50px;'>
    <p style='text-align:center; color:#AAAAAA; font-size:14px;'>
    Developed by <b style='color:#4B9CD3;'>Olawale Bello</b> | Data Science Intern, Wragby Business Solutions<br>
    <span style='font-size:12px;'>Built with ❤️ using Streamlit</span>
    </p>
""", unsafe_allow_html=True)






