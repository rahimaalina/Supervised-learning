import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import t

# DATASET
np.random.seed(42)
X = pd.DataFrame(np.random.randn(135, 7), columns=[f"x{i}" for i in range(1, 8)])
y = pd.Series(np.random.randint(1, 5, size=135))
X = (X - X.mean()) / X.std()

# PARAMETERS
lambdas = np.logspace(-4, 1, 5)
hidden_units = [1, 3, 5, 7, 10]
K1, K2 = 10, 10

# Outer CV
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Inner CV for Ridge Regression
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)
    best_lambda, best_ridge_error = None, float('inf')
    for lam in lambdas:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

            ridge = Ridge(alpha=lam)
            ridge.fit(X_inner_train, y_inner_train)
            inner_errors.append(mean_squared_error(y_inner_val, ridge.predict(X_inner_val)))

        avg_error = np.mean(inner_errors)
        if avg_error < best_ridge_error:
            best_ridge_error = avg_error
            best_lambda = lam

    ridge = Ridge(alpha=best_lambda)
    ridge.fit(X_train, y_train)
    ridge_test_error = mean_squared_error(y_test, ridge.predict(X_test))

    # INNER CV FOR ANN       ---- ASK ABT THIS
    best_h, best_ann_error = None, float('inf')
    for h in hidden_units:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train, X_inner_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

            ann = MLPRegressor(
                hidden_layer_sizes=(h,),
                max_iter=5000,
                learning_rate_init=0.01,
                solver='adam',
                random_state=42
            )
            ann.fit(X_inner_train, y_inner_train)
            inner_errors.append(mean_squared_error(y_inner_val, ann.predict(X_inner_val)))

        avg_error = np.mean(inner_errors)
        if avg_error < best_ann_error:
            best_ann_error = avg_error
            best_h = h

    ann = MLPRegressor(
        hidden_layer_sizes=(best_h,),
        max_iter=5000,
        learning_rate_init=0.01,
        solver='adam',
        random_state=42
    )
    ann.fit(X_train, y_train)
    ann_test_error = mean_squared_error(y_test, ann.predict(X_test))

    # BASELINE
    baseline_pred = np.mean(y_train)
    baseline_test_error = mean_squared_error(y_test, [baseline_pred] * len(y_test))

    results.append({
        "Fold": fold,
        "h* (ANN)": best_h,
        "E_test (ANN)": ann_test_error,
        "lambda* (Ridge)": best_lambda,
        "E_test (Ridge)": ridge_test_error,
        "E_test (Baseline)": baseline_test_error
    })

results_df = pd.DataFrame(results)

# Paired t-test for ANN, Ridge, and Baseline
ann_errors = results_df["E_test (ANN)"].values
ridge_errors = results_df["E_test (Ridge)"].values
baseline_errors = results_df["E_test (Baseline)"].values

def calculate_paired_ttest(data1, data2):
    differences = data1 - data2
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    p_value = 2 * t.cdf(-abs(t_stat), df=n-1)
    ci_lower = mean_diff - 1.96 * (std_diff / np.sqrt(n))
    ci_upper = mean_diff + 1.96 * (std_diff / np.sqrt(n))
    return mean_diff, ci_lower, ci_upper, p_value

#CALL FUNCT
# ANN vs Ridge
mean_ann_vs_ridge, ci_lower_ann_vs_ridge, ci_upper_ann_vs_ridge, p_value_ann_vs_ridge = calculate_paired_ttest(ann_errors, ridge_errors)
# ANN vs Baseline
mean_ann_vs_baseline, ci_lower_ann_vs_baseline, ci_upper_ann_vs_baseline, p_value_ann_vs_baseline = calculate_paired_ttest(ann_errors, baseline_errors)
# Ridge vs Baseline
mean_ridge_vs_baseline, ci_lower_ridge_vs_baseline, ci_upper_ridge_vs_baseline, p_value_ridge_vs_baseline = calculate_paired_ttest(ridge_errors, baseline_errors)

results_summary = {
    "Comparison": ["ANN vs Ridge", "ANN vs Baseline", "Ridge vs Baseline"],
    "Mean Difference": [mean_ann_vs_ridge, mean_ann_vs_baseline, mean_ridge_vs_baseline],
    "95% CI Lower": [ci_lower_ann_vs_ridge, ci_lower_ann_vs_baseline, ci_lower_ridge_vs_baseline],
    "95% CI Upper": [ci_upper_ann_vs_ridge, ci_upper_ann_vs_baseline, ci_upper_ridge_vs_baseline],
    "P-value": [p_value_ann_vs_ridge, p_value_ann_vs_baseline, p_value_ridge_vs_baseline]
}

results_summary_df = pd.DataFrame(results_summary)
print("Two-level CV Results")
print(results_df)
print("\nStatistical results")
print(results_summary_df)

# PLOT RESULTS
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc='center',
    cellLoc='center'
)
plt.title("Two-Llevel CV Results")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=results_summary_df.values,
    colLabels=results_summary_df.columns,
    loc='center',
    cellLoc='center'
)
plt.title("Statistical results")
plt.show()
