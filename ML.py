
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats

df = pd.read_csv(r"C:\Users\OlawaleBello\Documents\bank_data.csv")

from sklearn.model_selection import train_test_split


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train set: {train_set.shape}, Test set: {test_set.shape}")

df_train = train_set.copy()

df['was_previously_contacted'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)
df = pd.get_dummies(df, columns=['poutcome'], drop_first=True)
df = pd.get_dummies(df, columns=['contact'], drop_first=True)
df = pd.get_dummies(df, columns=['month'], drop_first=True)
df = pd.get_dummies(df, columns=['education'], drop_first=True)
df = df.drop('pdays', axis=1)
df = df.drop('Unnamed: 0', axis=1)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

y = df['y']
X = df.drop('y', axis=1)
y = y.apply(lambda x: 1 if x == 'yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
print(X.head())


scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') 

umerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features_to_encode = ['job', 'marital', 'default', 'housing', 'loan']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features_to_encode)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("Shapes of the resulting processed datasets:")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape:", X_test_processed.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier\
    
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_processed, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)

print("Logistic Regression, Random Forest, and Decision Tree models have been trained successfully.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Evaluation and parameter tuning
#Logistic Regression
log_reg_pred = log_reg_model.predict(X_test_processed)
log_reg_pred_proba = log_reg_model.predict_proba(X_test_processed)[:, 1] 

log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
log_reg_precision = precision_score(y_test, log_reg_pred)
log_reg_recall = recall_score(y_test, log_reg_pred)
log_reg_f1 = f1_score(y_test, log_reg_pred)
log_reg_roc_auc = roc_auc_score(y_test, log_reg_pred_proba)

print("Logistic Regression Model Performance:")
print(f"Accuracy: {log_reg_accuracy:.4f}")
print(f"Precision: {log_reg_precision:.4f}")
print(f"Recall: {log_reg_recall:.4f}")
print(f"F1-score: {log_reg_f1:.4f}")
print(f"ROC AUC: {log_reg_roc_auc:.4f}")
print("-" * 30)

rf_pred = rf_model.predict(X_test_processed)
rf_pred_proba = rf_model.predict_proba(X_test_processed)[:, 1] # Get probability of the positive class

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred) 
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_pred_proba)

print("Random Forest Model Performance:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-score: {rf_f1:.4f}")
print(f"ROC AUC: {rf_roc_auc:.4f}")
print("-" * 30)

dt_pred = dt_model.predict(X_test_processed)
dt_pred_proba = dt_model.predict_proba(X_test_processed)[:, 1] # Get probability of the positive class

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_roc_auc = roc_auc_score(y_test, dt_pred_proba)

print("Decision Tree Model Performance:")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall: {dt_recall:.4f}")
print(f"F1-score: {dt_f1:.4f}")
print(f"ROC AUC: {dt_roc_auc:.4f}")

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Define the parameter grid for Logistic Regression
log_reg_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'] # 'liblinear' supports both 'l1' and 'l2' penalties
}

# Define the parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
#Instantiate Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Instantiate GridSearchCV for Logistic Regression
log_reg_grid_search = GridSearchCV(estimator=log_reg, param_grid=log_reg_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit GridSearchCV to the training data
log_reg_grid_search.fit(X_train_processed, y_train)

# Print the best parameters and best score
print("Best parameters for Logistic Regression:")
print(log_reg_grid_search.best_params_)
print("\nBest cross-validation ROC AUC score for Logistic Regression:")
print(log_reg_grid_search.best_score_)

best_log_reg_model = log_reg_grid_search.best_estimator_


log_reg_tuned_pred = best_log_reg_model.predict(X_test_processed)


log_reg_tuned_pred_proba = best_log_reg_model.predict_proba(X_test_processed)[:, 1]

# Evaluate the tuned Logistic Regression model
log_reg_tuned_accuracy = accuracy_score(y_test, log_reg_tuned_pred)
log_reg_tuned_precision = precision_score(y_test, log_reg_tuned_pred)
log_reg_tuned_recall = recall_score(y_test, log_reg_tuned_pred)
log_reg_tuned_f1 = f1_score(y_test, log_reg_tuned_pred)
log_reg_tuned_roc_auc = roc_auc_score(y_test, log_reg_tuned_pred_proba)

print("Tuned Logistic Regression Model Performance on Test Set:")
print(f"Accuracy: {log_reg_tuned_accuracy:.4f}")
print(f"Precision: {log_reg_tuned_precision:.4f}")
print(f"Recall: {log_reg_tuned_recall:.4f}")
print(f"F1-score: {log_reg_tuned_f1:.4f}")
print(f"ROC AUC: {log_reg_tuned_roc_auc:.4f}")

rf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit GridSearchCV to the training data
rf_grid_search.fit(X_train_processed, y_train)

# Print the best parameters and best score
print("Best parameters for Random Forest:")
print(rf_grid_search.best_params_)
print("\nBest cross-validation ROC AUC score for Random Forest:")
print(rf_grid_search.best_score_)

best_rf_model = rf_grid_search.best_estimator_

# Make predictions on the test data
rf_tuned_pred = best_rf_model.predict(X_test_processed)

# Calculate predicted probabilities
rf_tuned_pred_proba = best_rf_model.predict_proba(X_test_processed)[:, 1]

# Evaluate the tuned Random Forest model
rf_tuned_accuracy = accuracy_score(y_test, rf_tuned_pred)
rf_tuned_precision = precision_score(y_test, rf_tuned_pred)
rf_tuned_recall = recall_score(y_test, rf_tuned_pred)
rf_tuned_f1 = f1_score(y_test, rf_tuned_pred)
rf_tuned_roc_auc = roc_auc_score(y_test, rf_tuned_pred_proba)

print("Tuned Random Forest Model Performance on Test Set:")
print(f"Accuracy: {rf_tuned_accuracy:.4f}")
print(f"Precision: {rf_tuned_precision:.4f}")
print(f"Recall: {rf_tuned_recall:.4f}")
print(f"F1-score: {rf_tuned_f1:.4f}")
print(f"ROC AUC: {rf_tuned_roc_auc:.4f}")

performance_comparison = pd.DataFrame({
    'Model': ['Logistic Regression (Original)', 'Logistic Regression (Tuned)',
              'Random Forest (Original)', 'Random Forest (Tuned)'],
    'Accuracy': [log_reg_accuracy, log_reg_tuned_accuracy, rf_accuracy, rf_tuned_accuracy],
    'Precision': [log_reg_precision, log_reg_tuned_precision, rf_precision, rf_tuned_precision],
    'Recall': [log_reg_recall, log_reg_tuned_recall, rf_recall, rf_tuned_recall],
    'F1-score': [log_reg_f1, log_reg_tuned_f1, rf_f1, rf_tuned_f1],
    'ROC AUC': [log_reg_roc_auc, log_reg_tuned_roc_auc, rf_roc_auc, rf_tuned_roc_auc]
})

# Display the comparison DataFrame
display(performance_comparison)

