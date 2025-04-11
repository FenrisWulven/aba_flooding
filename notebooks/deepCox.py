# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

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
# Nedbor is float64 but in 1,5 in mm, but it uses comma instead of dot. I want to use dot
print(f"Type of Nedbor: {df['Nedbor'].dtype}") # object
print(f"Type of Nedborsminutter: {df['Nedborsminutter'].dtype}") # object

# convert to a small memory float like float8
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
# check how many heavy rain events we have
print(f"Heavy rain events count: {df['heavy_rain'].sum()}")

######### Create a new dataframe with daily rainfall data I want 

# Create features for survival analysis
# We'll create time-to-event data where event is heavy rainfall
df = df.sort_values('datetime')

# Define durations (time to next heavy rain)
events = []
durations = []
features = []

# Set threshold for "heavy rain" event (in mm)
THRESHOLD = 5
# Window size for feature extraction (hours)
WINDOW = 24

print("Running feature extraction...")
for i in range(len(df) - WINDOW):
    # Extract window of data for feature calculation
    window = df.iloc[i:i+WINDOW]
    
    # Calculate features from this window
    total_rain = window['Nedbor'].sum()
    max_rain = window['Nedbor'].max()
    
    # Hour of day and day of year as cyclical features
    hour = window.iloc[-1]['datetime'].hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    day = window.iloc[-1]['datetime'].dayofyear
    day_sin = np.sin(2 * np.pi * day / 365)
    day_cos = np.cos(2 * np.pi * day / 365)
    
    # Time to next heavy rain event
    future_slice = df.iloc[i+WINDOW:]
    if len(future_slice) == 0:
        continue
    
    heavy_rain_idx = future_slice.index[future_slice['Nedbor'] > THRESHOLD].tolist()
    
    if heavy_rain_idx:  # If heavy rain occurs in the future
        next_rain_idx = heavy_rain_idx[0]
        duration = (future_slice.loc[next_rain_idx, 'datetime'] - window.iloc[-1]['datetime']).total_seconds() / 3600  # hours
        event = 1
    else:  # If no heavy rain in the remaining data
        duration = (df.iloc[-1]['datetime'] - window.iloc[-1]['datetime']).total_seconds() / 3600  # hours
        event = 0
    
    # Store the results
    features.append([total_rain, max_rain, hour_sin, hour_cos, day_sin, day_cos])
    durations.append(duration)
    events.append(event)

# Create dataset for survival analysis
survival_df = pd.DataFrame(features, columns=[
    'total_rain', 'max_rain', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
])
survival_df['duration'] = durations
survival_df['event'] = events

print(survival_df.head())

# Split the data into train, validation, and test sets
df_test = survival_df.sample(frac=0.2, random_state=1234)
df_train = survival_df.drop(df_test.index)
df_val = df_train.sample(frac=0.2, random_state=1234)
df_train = df_train.drop(df_val.index)

# Define columns for standardization
cols_standardize = ['total_rain', 'max_rain']
cols_leave = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']  

# Set up preprocessing with DataFrameMapper
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)

# Preprocess the data
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

# Extract target variables (duration and event)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)

# Define the neural network architecture
in_features = x_train.shape[1]  # Number of input features
num_nodes = [32, 32]            # Two hidden layers with 32 nodes each
out_features = 1                # Single output for CoxPH
batch_norm = True               # Use batch normalization
dropout = 0.1                   # Dropout rate to prevent overfitting
output_bias = False             # No bias in output layer

net = tt.practical.MLPVanilla(
    in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias
)

# Initialize the CoxPH model with the neural network and Adam optimizer
model = CoxPH(net, tt.optim.Adam)

# Set the learning rate for the optimizer
model.optimizer.set_lr(0.01)

# Define training parameters
epochs = 512                   # Maximum number of epochs
batch_size = 256               # Batch size for training
callbacks = [tt.callbacks.EarlyStopping()]  # Early stopping to prevent overfitting
verbose = True                 # Print training progress

# Train the model
log = model.fit(
    x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=(x_val, y_val)
)

# Compute baseline hazards
_ = model.compute_baseline_hazards()

# Predict survival curves for the test set
surv = model.predict_surv_df(x_test)

# Evaluate model performance using the concordance index
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
concordance = ev.concordance_td('antolini')
print(f"Concordance index: {concordance}")

# Plot survival curves
plt.figure(figsize=(12, 8))
for i in range(min(10, len(surv.columns))):  # Plot the first 10 samples
    plt.step(surv.index, surv.iloc[:, i], where="post", label=f"Sample {i}")
plt.xlabel('Time (hours)')
plt.ylabel('Probability of no heavy rainfall')
plt.grid(True)
plt.title('Survival Curves: Probability of Avoiding Heavy Rainfall')
plt.legend()
plt.savefig('rainfall_survival_curves.png')
plt.show()

# Plot mean survival curve
plt.figure(figsize=(10, 6))
mean_surv = surv.mean(axis=1)
plt.step(surv.index, mean_surv, where="post", linewidth=2)
plt.xlabel('Time (hours)')
plt.ylabel('Mean probability of no heavy rainfall')
plt.grid(True)
plt.title('Mean Survival Curve for Heavy Rainfall Events')
plt.savefig('mean_rainfall_survival_curve.png')
plt.show()

# Calculate and plot feature importance
feature_names = cols_standardize + cols_leave
feature_importance = np.abs(model.net.parameters()[-2].detach().numpy()).flatten()

# Normalize importance values
feature_importance = feature_importance / np.sum(feature_importance)

# Create bar plot for feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Rainfall Prediction')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rainfall_feature_importance.png')
plt.show()
