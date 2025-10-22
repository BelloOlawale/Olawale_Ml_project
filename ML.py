
import pandas as pd
import numpy as np
import os
import joblib
from imblearn.over_sampling import SMOTE
save_path = r"C:\Users\OlawaleBello\Documents"
os.makedirs(save_path, exist_ok=True)


df = pd.read_csv(r"C:\Users\OlawaleBello\Documents\bank_data.csv")

df.head()

#drop the column Unnamed

df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

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

# Use a single final logistic regression model with class balancing
log_reg = LogisticRegression(
    class_weight='balanced',     # penalize minority underrepresentation
    solver='liblinear',          # stable for small/medium datasets
    max_iter=1000,
    random_state=42
)

# Train final logistic regression model
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
# Save artifacts locally (project root)
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")


print(f"✅ Model, scaler, and columns saved successfully to: {save_path} and project root.")




