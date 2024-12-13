import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

#DATASET
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
print("Initial Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# MISSING VALUES ##########
X = X.fillna(X.mean())
print("\nAfter filling missing v:")
print(X.isnull().sum())

#STD+MEAN
categorical_columns = ["cp", "restecg", "slope", "ca", "thal"]
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print("\nAfter One-Hot Encoding:")
print(X.head())
print(f"New Shape of Features: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nAfter Standardization:")
print(f"Mean: {np.mean(X_scaled, axis=0)}")
print(f"Standard deviation {np.std(X_scaled, axis=0)}")
