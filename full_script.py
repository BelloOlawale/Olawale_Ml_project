import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

f = pd.read_csv(r"C:\Users\OlawaleBello\Documents\bank_data.csv")
df.head()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()

display(df.describe(include = ['object', 'bool', 'category']).T)
df.loc[df.duplicated()]
df=df.copy()

def summary_numerical_dist(df_data, col, q_min, q_max):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (2, 2)
    num_subplot = 4
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 1, 1], [1, 1, 1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )

    sns.histplot(data = df_data, x = col, kde = True, ax = axes[0])
    stats.probplot(x = df_data[col], dist = stats.norm, plot = axes[1])
    sns.boxplot(data = df_data, x = col, ax = axes[2])
    pts = df_data[col].quantile(q = np.arange(q_min, q_max, 0.01))
    sns.lineplot(x = pts.index, y = pts, ax = axes[3])
    axes[3].grid(True)

    list_title = ["Histogram", "QQ plot", "Boxplot", "Outlier"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()
    
    ef summary_categorical_dist(df_data, col):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (1, 2)
    num_subplot = 2
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1)], [1, 1], [1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )
    
    count = df_data[col].value_counts().sort_index()
    
    sns.countplot(data = df_data, y = col, order = count.index, ax = axes[0])
    axes[1].pie(data = df_data, x = count, labels = count.index, autopct = '%1.1f%%', startangle = 90)
    list_title = ["Counts", "Proportions"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()
    
    summary_numerical_dist(df, 'age', .95, 1)
    
    summary_numerical_dist(df, 'balance', .95, 1)
    
    summary_numerical_dist(df, 'duration', .95, 1)
    
    summary_numerical_dist(df, 'day', .95, 1)
    
    summary_numerical_dist(df, 'campaign', .95, 1)
    
    summary_numerical_dist(df, 'pdays', .95, 1)
    
    summary_numerical_dist(df, 'previous', .95, 1)
    
    # plotting histograms for all numerical features
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()

# Unique values in each categorical column
for col in df.select_dtypes(include='number').columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(), "\n")

from pandas.plotting import scatter_matrix
attributes = [
    'duration', 'age', 'day', 'pdays',
    'campaign', 'previous',
]

scatter_matrix(df[attributes], figsize=(20, 15))
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribution of subscription')
plt.xlabel('age')
plt.ylabel('subscribed')
plt.show()

summary_categorical_dist(df, 'y')
summary_categorical_dist(df, 'job')
summary_categorical_dist(df, 'default')
summary_categorical_dist(df, 'education')
summary_categorical_dist(df, 'marital')
summary_categorical_dist(df, 'housing')
summary_categorical_dist(df, 'loan')
summary_categorical_dist(df, 'contact')
summary_categorical_dist(df, 'month')
summary_categorical_dist(df, 'poutcome')
summary_categorical_dist(df, 'day_of_week')

# Unique values in each categorical column
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(), "\n")

y = df['y']              # Target column
X = df.drop('y', axis=1) # All other columns as features

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train set: {train_set.shape}, Test set: {test_set.shape}")

df_train = train_set.copy()
#drop the column Unnamed

df.drop('Unnamed: 0', axis=1, inplace=True)

#getting the data types for the dataset


df.dtypes

# split table in categorcal and numnerical columns

categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

numerical_columns

categorical_columns

"""# Random forest Part"""

from sklearn.preprocessing import LabelEncoder

# Copy dataframe
df_tree = df.copy()

# Apply Label Encoding to categorical columns
le = LabelEncoder()
for col in categorical_columns:
    df_tree[col] = le.fit_transform(df_tree[col])

# Separate target variable
target_col = 'y'
X_tree = df_tree.drop(target_col, axis=1)
y_tree = df_tree[target_col]

#random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score



df_tree.head()

#split data

X_train, X_test, y_train, y_test = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

#train model

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# evaluate

y_pred = rf_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#class-weight balanced to handle imbalaqnce


rf_balanced = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')

#Train
rf_balanced.fit(X_train, y_train)

#predict

y_pred_balanced = rf_balanced.predict(X_test)
y_proba_balanced = rf_balanced.predict_proba(X_test)[:, 1]

#evaluate

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("\nClassification Report:\n", classification_report(y_test, y_pred_balanced))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba_balanced))

"""##Using SMOTE"""

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_smote.fit(X_res, y_res)

y_pred_smote = rf_smote.predict(X_test)
y_proba_smote = rf_smote.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("ROC AUC:", roc_auc_score(y_test, y_proba_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))

"""## logistic regression"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate target column
target_col = 'y'
categorical_features = [col for col in categorical_columns if col != target_col]

# One-hot encode
df_log = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Convert target to numeric (0/1)
df_log[target_col] = df_log[target_col].map({'yes': 1, 'no': 0})

# Split features & target
X = df_log.drop(target_col, axis=1)
y = df_log[target_col]

X.head()

# Scale numerical columns (important for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression

# Initialize model with class balancing
log_reg = LogisticRegression(
    class_weight='balanced',     # penalize minority underrepresentation
    solver='liblinear',          # stable for small/medium datasets
    max_iter=1000,
    random_state=42
)

# Train
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

# Evaluate
print("Accuracy:", log_reg.score(X_test, y_test))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

# Save model
joblib.dump(log_reg, "logistic_regression_model.pkl")

# Optionally save the scaler too (since you scaled features before training)
joblib.dump(scaler, "scaler.pkl")

print("✅ Logistic Regression model and scaler saved successfully!")





import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Lazy-load model at prediction time to avoid heavy imports / DLL errors on import
model = None
scaler = None
model_columns = None
MODEL_PATH = "logistic_regression_model.pkl"
SCALER_PATH = "scaler.pkl"
MODEL_COLUMNS_PATH = "model_columns.pkl"

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
    # declare we will assign to these module-level objects
    global model, scaler, model_columns
    # Try to load the trained model lazily (this triggers sklearn/scipy imports)
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            st.error(f"Model file '{MODEL_PATH}' not found in project root. Run training (ML.py) to create it.")
            st.stop()
        except ImportError as ie:
            st.error("A dependency required to load the model is missing (e.g. scipy or scikit-learn).\n"
                     "Install them in your environment: `pip install scipy scikit-learn` and restart the app.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

    # Before predicting, prepare features the same way as during training
    # Build a DataFrame with raw inputs (categorical as strings)
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

    # Load scaler and model_columns lazily if not already loaded
    if scaler is None or model_columns is None:
        try:
            scaler = joblib.load(SCALER_PATH)
            model_columns = joblib.load(MODEL_COLUMNS_PATH)
        except FileNotFoundError as fnf:
            st.error(f"Required artifact not found: {fnf.filename}. Run ML.py to generate artifacts.")
            st.stop()
        except ImportError:
            st.error("A dependency required to load the scaler/model columns is missing. Install scipy/scikit-learn.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load preprocessing artifacts: {e}")
            st.stop()

    # One-hot encode and align columns
    try:
        input_encoded = pd.get_dummies(input_raw, drop_first=True)
        # model_columns may be stored as list-like
        input_aligned = input_encoded.reindex(columns=list(model_columns), fill_value=0)
        scaled_input = scaler.transform(input_aligned)
    except Exception as e:
        st.error(f"Failed to preprocess input for the model: {e}")
        st.stop()
# Attempt prediction
try:
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]
    
    if prediction[0] == 1:
        st.success(f"✅ The customer is likely to SUBSCRIBE! (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ The customer is unlikely to subscribe. (Confidence: {1 - probability:.2%})")
        
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
    st.stop()
    # Attempt prediction
    
# Footer
st.markdown("""
    <hr style='margin-top: 50px;'>
    <p style='text-align:center; color:#AAAAAA; font-size:14px;'>
    Developed by <b style='color:#4B9CD3;'>Olawale Bello</b> | Data Science Intern, Wragby Business Solutions<br>
    <span style='font-size:12px;'>Built with ❤️ using Streamlit</span>
    </p>
""", unsafe_allow_html=True)






