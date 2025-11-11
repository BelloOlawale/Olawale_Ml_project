"""Simple harness to run the Streamlit app's prediction pipeline outside Streamlit.
This mirrors the preprocessing in `streamlitapi.py` so we can debug model/scaler/columns issues.
"""
import joblib
import pandas as pd
import numpy as np
import sys

MODEL = 'logistic_regression_model.pkl'
SCALER = 'scaler.pkl'
COLUMNS = 'model_columns.pkl'

# sample inputs (match UI choices)
sample = {
    'age': 30,
    'balance': 1000,
    'duration': 100,
    'campaign': 2,
    'pdays': -1,
    'previous': 0,
    'job': 'services',
    'marital': 'married',
    'education': 'secondary',
    'housing': 'yes',
    'loan': 'no',
    'poutcome': 'unknown'
}

try:
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    model_columns = joblib.load(COLUMNS)
except Exception as e:
    print('ERROR loading artifacts:', e)
    sys.exit(1)

print('Loaded model type:', type(model))
print('Loaded scaler type:', type(scaler))
print('Model columns count:', len(list(model_columns)))

# Build DataFrame
input_raw = pd.DataFrame([sample])
print('\nRaw input:')
print(input_raw.to_dict(orient='records')[0])

# One-hot encode and align
input_encoded = pd.get_dummies(input_raw, drop_first=True)
input_aligned = input_encoded.reindex(columns=list(model_columns), fill_value=0)

print('\nEncoded sample (sparse):')
print(input_aligned.head().to_string())

# Scale
try:
    X = scaler.transform(input_aligned)
except Exception as e:
    print('ERROR scaling input:', e)
    sys.exit(1)

print('\nScaled shape:', X.shape)

# Predict
try:
    pred = model.predict(X)
    proba = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
    print('\nPrediction:', pred)
    print('Probability (class 1):', proba)
except Exception as e:
    print('ERROR predicting:', e)
    sys.exit(1)

print('\nSUCCESS: end-to-end prediction completed')
