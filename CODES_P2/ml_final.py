import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Define features (X) and target (y)
X = heart_disease.data.features[['age', 'cp', 'chol', 'thalach']]  
y = (heart_disease.data.targets['num'] > 0).astype(int)  # Binary classification: 0 (no disease), 1 (disease)

# Normalize continuous features
scaler = StandardScaler()
X[['age', 'chol', 'thalach']] = scaler.fit_transform(X[['age', 'chol', 'thalach']])

# Baseline Model: Predicts the majority class
def baseline_model(y_train, y_test):
    majority_class = y_train.value_counts().idxmax()
    y_pred = [majority_class] * len(y_test)
    return accuracy_score(y_test, y_pred)

# Two-level cross-validation
kf_outer = KFold(n_splits=10, shuffle=True, random_state=42)
log_reg_errors = []
dt_errors = []
baseline_errors = []

for train_index, test_index in kf_outer.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Inner cross-validation for parameter tuning
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Logistic Regression parameter tuning
    best_log_reg_model = None
    best_log_reg_score = float('-inf')
    for C in [0.01, 0.1, 1, 10]:  # Regularization values
        log_reg = LogisticRegression(C=1/C, solver='lbfgs', max_iter=1000)
        scores = cross_val_score(log_reg, X_train, y_train, cv=kf_inner, scoring='accuracy')
        if scores.mean() > best_log_reg_score:
            best_log_reg_score = scores.mean()
            best_log_reg_model = log_reg
    
    # Decision Tree parameter tuning
    best_dt_model = None
    best_dt_score = float('-inf')
    for max_depth in [2, 3, 4, 5]:  
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        scores = cross_val_score(dt, X_train, y_train, cv=kf_inner, scoring='accuracy')
        if scores.mean() > best_dt_score:
            best_dt_score = scores.mean()
            best_dt_model = dt
    
    # Train the best models on the outer fold training data
    best_log_reg_model.fit(X_train, y_train)
    best_dt_model.fit(X_train, y_train)
    
    # Evaluate on the outer fold test data
    log_reg_pred = best_log_reg_model.predict(X_test)
    dt_pred = best_dt_model.predict(X_test)
    
    log_reg_errors.append(1 - accuracy_score(y_test, log_reg_pred))  # Misclassification rate
    dt_errors.append(1 - accuracy_score(y_test, dt_pred))
    baseline_errors.append(1 - baseline_model(y_train, y_test))

# Create a summary table of errors for each fold
results = pd.DataFrame({
    'Outer Fold': range(1, 11),
    'Logistic Regression Error': log_reg_errors,
    'Decision Tree Error': dt_errors,
    'Baseline Error': baseline_errors
})

print(results)
print("Average Errors:")
print(results.mean())

# Paired t-tests between models
log_reg_vs_dt = ttest_rel(log_reg_errors, dt_errors)
log_reg_vs_baseline = ttest_rel(log_reg_errors, baseline_errors)
dt_vs_baseline = ttest_rel(dt_errors, baseline_errors)

print("Logistic Regression vs Decision Tree: t-statistic = {:.3f}, p-value = {:.3f}".format(log_reg_vs_dt.statistic, log_reg_vs_dt.pvalue))
print("Logistic Regression vs Baseline: t-statistic = {:.3f}, p-value = {:.3f}".format(log_reg_vs_baseline.statistic, log_reg_vs_baseline.pvalue))
print("Decision Tree vs Baseline: t-statistic = {:.3f}, p-value = {:.3f}".format(dt_vs_baseline.statistic, dt_vs_baseline.pvalue))


data = heart_disease.data.features 
# Define features (X) and target (y)
X = data[['age', 'cp', 'chol', 'thalach']]  # Example features
y = (heart_disease.data.targets > 0).astype(int)  # Binary target: 0 (no disease), 1 (disease)

# Normalize continuous features
scaler = StandardScaler()
X[['age', 'chol', 'thalach']] = scaler.fit_transform(X[['age', 'chol', 'thalach']])

# Train Logistic Regression using the best lambda
best_lambda = 0.1  # Replace with the value from cross-validation
C = 1 / best_lambda
log_reg = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
log_reg.fit(X, y)

# Extract feature importance (coefficients)
coefficients = pd.DataFrame({
    'Feature': ['age', 'cp', 'chol', 'thalach'],
    'Coefficient': log_reg.coef_[0]
})

print("Feature Coefficients:")
print(coefficients)

# Example prediction
example_patient = np.array([[60, 3, 240, 140]])  # Replace with patient data
example_patient[:, [0, 2, 3]] = scaler.transform(example_patient[:, [0, 2, 3]])  # Normalize
probability = log_reg.predict_proba(example_patient)[0, 1]  # Probability of disease presence
print(f"Predicted probability of disease presence: {probability:.3f}")