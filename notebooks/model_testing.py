import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/Regn2020-2025.csv",sep=";")
df['Nedborsminutter'] = df['Nedborsminutter'].fillna(0.0)

df_only_rain = df[df['Nedborsminutter'] > 0.5]

## Models to test:
# 2. GAN (Generative Adversarial Network)
# Use the timeseries data
# 3. Isolation Forest (Anomaly detection)
# Use the timeseries data
# 4. Transformer model
# Temporal Fusion Transformers (TFT) or similar
# 5. Gaussian Processes
# 6. Quantile Regression Models (QRM)
# 7. ConvLSTM (Convolutional LSTM)

# 2. GAN (Generative Adversarial Network)

# 3. Isolation Forest (Anomaly detection) Not really useful for predictions
# Isolation forest seems to only be able to capture an actual rain event as that counts as the anomoly, 
# perhaps if combined with another model
data = df['Nedborsminutter']
model = IsolationForest()

model.fit(data.values.reshape(-1,1))
df['anomaly'] = model.predict(data.values.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(df['Nedborsminutter'],label='Nedborsminutter')
plt.plot(df['anomaly'],label='Anomaly')
plt.savefig("isolation_forest.png")
plt.close()

data = df_only_rain['Nedborsminutter']
model = IsolationForest()

model.fit(data.values.reshape(-1,1))
df_only_rain['anomaly'] = model.predict(data.values.reshape(-1,1))

plt.figure(figsize=(10,6))
plt.plot(df_only_rain['Nedborsminutter'],label='Nedborsminutter')
plt.fill_between(df_only_rain.index, df_only_rain['Nedborsminutter'].min(), 
                 df_only_rain['Nedborsminutter'].max(), 
                 where=df_only_rain['anomaly'] == -1, color='orange', alpha=0.3, label='Anomaly')
plt.savefig("isolation_forest_only_rain.png")
plt.close()

# 4. Transformer model
# Temporal Fusion Transformers (TFT) or similar



# 5. Gaussian Processes
data = df['Nedborsminutter']

# 6. Quantile Regression Models (QRM)
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = df['Nedborsminutter']
df_ts = df.copy()
df_ts['time_idx'] = np.arange(len(df_ts))  # Add time index as a feature

# Create lag features for time series forecasting
for i in range(1, 4):  # Creating 3 lag features
    df_ts[f'lag_{i}'] = df_ts['Nedborsminutter'].shift(i)

# Drop NaNs created by lag features
df_ts = df_ts.dropna()

# Features and target
X = df_ts[['time_idx'] + [f'lag_{i}' for i in range(1, 4)]]
y = df_ts['Nedborsminutter']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit models for different quantiles
quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
quantile_models = {}

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.5)
    model.fit(X_train_scaled, y_train)
    quantile_models[q] = model

# Predict on test set
y_pred = {}
for q, model in quantile_models.items():
    y_pred[q] = model.predict(X_test_scaled)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, 'k-', label='Actual')

colors = ['blue', 'green', 'orange', 'red', 'purple']
for (q, pred), color in zip(y_pred.items(), colors):
    plt.plot(y_test.index, pred, color=color, linestyle='-', label=f'Q{q}')

plt.legend()
plt.title('Quantile Regression Predictions for Rainfall')
plt.xlabel('Time')
plt.ylabel('Nedborsminutter')
plt.savefig("quantile_regression.png")
plt.close()

# Plot focusing on periods with significant rainfall
high_rain_idx = y_test > y_test.quantile(0.9)
if high_rain_idx.any():
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index[high_rain_idx], y_test.values[high_rain_idx], 'k-', label='Actual')
    
    for (q, pred), color in zip(y_pred.items(), colors):
        plt.plot(y_test.index[high_rain_idx], pred[high_rain_idx], color=color, linestyle='-', label=f'Q{q}')
    
    plt.legend()
    plt.title('Quantile Regression Predictions for High Rainfall Periods')
    plt.xlabel('Time')
    plt.ylabel('Nedborsminutter')
    plt.savefig("quantile_regression_high_rain.png")
    plt.close()


# 7. ConvLSTM (Convolutional LSTM)
