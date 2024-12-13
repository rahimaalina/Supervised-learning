import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo


heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

##MISSING
X = X.fillna(X.mean())

#One-hot
categorical_columns = ["cp", "restecg", "slope", "ca", "thal"]
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


original_first_observation = X.iloc[0]
print("\nOriginal feature values of the 1 obs:")
print(original_first_observation)
lambdas = np.logspace(-4, 3, 50)  
generalization_errors = []

#  K=10 fold cv       #############
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for lam in lambdas:
    ridge = Ridge(alpha=lam)  
    fold_errors = []  

    for train_index, test_index in kf.split(X_scaled):

        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

       
        fold_errors.append(mean_squared_error(y_test, y_pred))
    
    
    generalization_errors.append(np.mean(fold_errors))

#PLOT
plt.figure(figsize=(8, 6))
plt.plot(lambdas, generalization_errors, marker='o')
plt.xscale("log")  
plt.xlabel("Regularization p")
plt.ylabel("Generalization errorr")
plt.title("Estimated generalization error as a function of λ")
plt.grid(True)
plt.show()

###### JUST TO CHECK
best_lambda = lambdas[np.argmin(generalization_errors)]
print(f"Best lambda: {best_lambda}")
best_ridge = Ridge(alpha=best_lambda)
best_ridge.fit(X_scaled, y)
coefficients = best_ridge.coef_.flatten() if len(best_ridge.coef_.shape) > 1 else best_ridge.coef_
columns = list(X.columns)
feature_importance = pd.DataFrame({
    "Feature": columns,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)
print("\nFeature Importance:")
print(feature_importance)
lowest_error = min(generalization_errors)
print(f"Lowest eerror: {lowest_error}")
example_x = X_scaled[0] 
predicted_y = best_ridge.predict([example_x])
print(f"Predicted y for the first observation: {predicted_y[0]}")
