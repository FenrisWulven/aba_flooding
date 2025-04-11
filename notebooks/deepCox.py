# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# Load the rainfall dataset
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
DATAFILE = os.path.join(parent_parent_dir, 'data', 'raw', 'Regn2004-2025.csv')
try:    
    df = pd.read_csv(DATAFILE, sep=';', decimal=',')
    print(f"Data Successfully loaded from {DATAFILE}")
except FileNotFoundError:
    print(f"File not found: {DATAFILE}")
    raise

df['Dato'] = pd.to_datetime(df['Dato'], format='%d.%m.%Y')
df['Tid'] = df.apply(lambda x: int(x['Tid'].split(':')[0]), axis=1) # Convert time to hours e.g. '12:00:00' to 12
print(f"First row of time: {df['Tid'].iloc[0]}")
df['datetime'] = df['Dato'] + pd.to_timedelta(df['Tid'], unit='h') 
print(f"First rows of datetime: {df['datetime'].iloc[0]}")

# Print type of nedbor and nedborsminutter columns
print(f"Type of Nedbor: {df['Nedbor'].dtype}") # object
print(f"Type of Nedborsminutter: {df['Nedborsminutter'].dtype}") # object

# Convert to a small memory float like float8
df['Nedbor'] = df['Nedbor'].astype('float16') # Convert to float16
df['Nedborsminutter'] = df['Nedborsminutter'].astype('float16') # Convert to float16

print(f"Type of Nedbor: {df['Nedbor'].dtype}")
print(f"Type of Nedborsminutter: {df['Nedborsminutter'].dtype}") 

########## NAN ########
# Check nans in Nedbor and Nedborsminutter columns
print(f"Nedbor NaN count: {df['Nedbor'].isna().sum()}")
print(f"Nedborsminutter NaN count: {df['Nedborsminutter'].isna().sum()}")
# Indexes of the Nans 
nedbor_nan_indexes = df[df['Nedbor'].isna()].index.tolist()
nedborsminutter_nan_indexes = df[df['Nedborsminutter'].isna()].index[:2].tolist()

print(f"Nedbor NaN indexes: {nedbor_nan_indexes[:2]}")
print(f"Nedborsminutter NaN indexes: {nedborsminutter_nan_indexes[:2]}")
# Show the first 2 Nana Nedbor and Nedborsminutter values
print(f"First 2 Nedbor nan values: {df['Nedbor'].iloc[nedbor_nan_indexes[:2]].tolist()}")
print(f"First 2 Nedborsminutter nan values: {df['Nedborsminutter'].iloc[nedborsminutter_nan_indexes[:2]].tolist()}")

df['Nedbor'] = df['Nedbor'].fillna(0)
df['Nedborsminutter'] = df['Nedborsminutter'].fillna(0)

#### Preprocess the dataset
df['heavy_rain'] = (df['Nedbor'] > 5)  # Define "event" as rainfall > 5mm
# Check how many heavy rain events we have
print(f"Heavy rain events count: {df['heavy_rain'].sum()}")

# Find the time beween heavy rain events
df['TTE'] = df['heavy_rain'].astype(int).diff().fillna(0).cumsum()
print(f"First row of TTE: {df['TTE'].iloc[0]}")

######### Create a new dataframe with daily rainfall data I want 

# Create features for survival analysis
# We'll create time-to-event data where event is heavy rainfall
df = df.sort_values('datetime')
print(f"First rows of the sorted dataframe: \n{df.head()}")

# # Aggregate by day
# def process_day(group):
#     # Find first heavy_rain event
#     event_rows = group[group['heavy_rain'] == True]
#     if not event_rows.empty:
#         # Time to first event
#         duration = event_rows.iloc[0]['Tid']
#         event = 1
#         # Use features just before or at event
#         features = group[group['Tid'] <= duration][['Nedbor', 'Nedborsminutter']].mean()
#     else:
#         # Time to the end of the day (last recorded hour)
#         duration = 24
#         event = 0
#         features = group[['Nedbor', 'Nedborsminutter']].mean()
    
#     return pd.Series({
#         'duration': duration,
#         'event': event,
#         'Nedbor': features['Nedbor'],
#         'Nedborsminutter': features['Nedborsminutter']
#     })

# # Apply to each day
# survival_data = df.groupby('Dato').apply(process_day).reset_index()
# # save as csv in data/processed
# survival_data.to_csv(os.path.join(parent_parent_dir, 'data', 'processed', 'survival_data.csv'), index=False)
# print("Survival data saved to CSV.")
# print(survival_data.head())

# # Load the survival data
# survival_data = pd.read_csv(os.path.join(parent_parent_dir, 'data', 'processed', 'survival_data.csv'))

# ==== Deep Cox PH model ====

# Extract features and targets
cols = ['Nedbor', 'Nedborsminutter']
x_raw = df[cols].values
scaler = StandardScaler()
x = scaler.fit_transform(x_raw).astype('float32')  # Explicitly cast to float32
# events columns is the heavy rain event boolean
events = df['heavy_rain'].values  # Event indicator (1 if heavy rain, 0 otherwise)
# Convert boolean to int (0 or 1)
events = events.astype(int)
# TTE is the duration column
durations = df['Tid'].values  # Time to event in hours

# Build the neural network architecture for the Deep Cox PH model
in_features = x.shape[1]
hidden_layers = [32, 32]  # you can adjust hidden layer sizes
net = tt.practical.MLPVanilla(in_features, hidden_layers, 1, dropout=0.1)

# Determine device and move the model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Instantiate the Cox PH model with the neural network
model = CoxPH(net, optimizer=torch.optim.Adam)

# Train the model (the fit method will use the device set in the network)
model.fit(x, (durations, events), epochs=20, batch_size=32, verbose=True)
print("Deep Cox PH model training complete.")

# save the model
model_path = os.path.join(parent_parent_dir, 'models', 'deep_cox_model.pt')
model.save(model_path)
print(f"Model saved to {model_path}")


# Plot survival curves
surv = model.predict_surv_df(x)
time_points = surv.index
plt.figure(figsize=(10, 6))
for i in range(min(10, len(x))):  # Plot survival curves for the first 10 samples
    plt.step(time_points, surv.iloc[:, i], where="post", label=f"Sample {i+1}")
plt.title("Survival Curves")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True)
plt.show()

