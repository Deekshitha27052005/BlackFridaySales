import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

# Load data
df = pd.read_csv("BlackFridaySales.csv")

# ---------------------------
# Data Cleaning
# ---------------------------

# Fill missing values (FIXED - no inplace warning)
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

# Encode categorical columns
cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Drop IDs (ignore if not present)
df = df.drop(['User_ID', 'Product_ID'], axis=1, errors='ignore')

# ---------------------------
# Train-Test Split
# ---------------------------

X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# XGBoost Model
# ---------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------

y_pred = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nBlack Friday Sales Prediction")
print("--------------------------------")
print("RMSE:", rmse)