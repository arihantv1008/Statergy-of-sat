# Program 3: Time Series, Data Vectors & Forecasting using Linear Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create DataFrame with Date, Sales, Temperature
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-31', periods=4, freq='M'),   # monthly dates
    'Sales': [100, 120, 140, 160],                              # sales values
    'Temp': [25, 26, 28, 30]                                    # temperature values
})

# Display the full DataFrame
print(df)

# Display sales as a single vector
print("Sales vector:", df['Sales'].values)

# Display multivariate vector (Sales + Temp)
print("Multivariate:", df[['Sales', 'Temp']].values)

# Create time index for regression (0,1,2,3)
df['t'] = np.arange(len(df))

# Build linear regression model
m = LinearRegression()
m.fit(df[['t']], df['Sales'])   # Train model using time vs sales

# Predict next 2 time steps (t = 4, 5)
future = np.arange(len(df), len(df) + 2).reshape(-1, 1)
print("Forecast next 2:", m.predict(future))
