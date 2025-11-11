import joblib
import sys
try:
    m = joblib.load(r'c:\Users\OlawaleBello\workspace\Olawale_Ml_project\logistic_regression_model.pkl')
    s = joblib.load(r'c:\Users\OlawaleBello\workspace\Olawale_Ml_project\scaler.pkl')
    c = joblib.load(r'c:\Users\OlawaleBello\workspace\Olawale_Ml_project\model_columns.pkl')
    print('model type:', type(m), 'has predict:', hasattr(m,'predict'), 'has predict_proba:', hasattr(m,'predict_proba'))
    print('scaler type:', type(s))
    print('columns len:', len(list(c)))
except Exception as e:
    print('ERROR loading artifacts:', e)
    sys.exit(1)
