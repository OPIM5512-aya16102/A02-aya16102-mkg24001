from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

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

#Test, 10% in validation and 20% in test
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)