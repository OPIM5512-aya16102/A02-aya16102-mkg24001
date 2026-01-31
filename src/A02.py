# Load Packages
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # keep output clean; MLP may warn about convergence
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

#Train/Test split
#Separate Features (X) and Target (y)
x = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

#Split the data
#Train, 20% goes into test pile
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Test, 10% in validation and 10% in test
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# Adding MLPRegressor with early stopping

# Step 1: Lets Scale the features:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_val_scaled   = scaler.transform(x_val)
X_test_scaled  = scaler.transform(x_test)

# Fit the model
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10,5),
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) # important!
mlp.fit(X_train_scaled, y_train)

print(X_train_scaled)

# Step 2: Predict on all splits
y_pred_train = mlp.predict(X_train_scaled)
y_pred_val   = mlp.predict(X_val_scaled)
y_pred_test  = mlp.predict(X_test_scaled)
 
# Step 3: Scatterplots: predicted vs actual (one figure per split; y=x reference line)
def scatter_with_reference(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')  # reference line
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()
    plt.show()

scatter_with_reference(y_train, y_pred_train, "Predicted vs Actual — Train")
# Save the plot
plt.savefig("figures/train_actual_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()
scatter_with_reference(y_test,  y_pred_test,  "Predicted vs Actual — Test")
plt.savefig("figures/test_actual_vs_pred.png", dpi=300, bbox_inches="tight")