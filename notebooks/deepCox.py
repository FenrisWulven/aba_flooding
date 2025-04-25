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

# Calculate Time To Event (TTE) properly
# First, get indices where heavy rain events occur
event_indices = df.index[df['heavy_rain'] == True].tolist()
# Add last index to handle observations after the last event
event_indices.append(len(df))

# Initialize TTE column with a large value
df['TTE'] = len(df)  # Default to maximum possible time

# For each observation, calculate time to the next event
for i in range(len(event_indices)-1):
    current_event = event_indices[i]
    next_event = event_indices[i+1]
    
    # For all observations between current event and next event
    for j in range(current_event, next_event):
        df.loc[j, 'TTE'] = next_event - j
        
# For observations with heavy_rain=True, set TTE to 0
df.loc[df['heavy_rain'] == True, 'TTE'] = 0


print(f"First heavy rain index: {event_indices[0]}")
# head starting from the first heavy rain event
print(f"First rows of the dataframe starting from the first heavy rain event: \n{df.iloc[event_indices[0]-1:event_indices[0]+4]}")

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
# TTE
durations = df['TTE'].values  # Time to event (TTE) in hours



# Build the neural network architecture for the Deep Cox PH model
in_features = x.shape[1]
hidden_layers = [32, 32]  # you can adjust hidden layer sizes
net = tt.practical.MLPVanilla(in_features, hidden_layers, 1, dropout=0.1)

# Determine device and move the model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net.to(device)

### Training the model
# Instantiate the Cox PH model with the neural network
model = CoxPH(net, optimizer=torch.optim.Adam)

# Train the model (the fit method will use the device set in the network)
model.fit(x, (durations, events), epochs=2, batch_size=32, verbose=True)
print("Deep Cox PH model training complete.")

# Compute baseline hazards before predicting
model.compute_baseline_hazards()

H0 = model.baseline_cumulative_hazards_   # pandas Series indexed by time
S0 = np.exp(-H0)                          # baseline survival = exp(–cumhaz)

plt.figure(figsize=(8,5))
plt.step(S0.index, S0, where='post')
plt.title("Baseline Survival Curve")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(os.path.join(parent_parent_dir, 'reports', 'figures', 'baseline_survival_curve.png'))
plt.show()

# Plot survival curves
surv = model.predict_surv_df(x)
time_points = surv.index

# plt.figure(figsize=(10, 6))
# for i in range(min(10, len(x))):  # Plot survival curves for the first 10 samples
#     plt.step(time_points, surv.iloc[:, i], where="post", label=f"Sample {i+1}")
# plt.title("Survival Curves")
# plt.xlabel("Time")
# plt.ylabel("Survival Probability")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(parent_parent_dir, 'reports', 'figures', 'survival_curves.png'))
# plt.show()

model_path = os.path.join(parent_parent_dir, 'models', 'deep_cox_model.pth')
# Instead of model.save(), save the network's state_dict


torch.save(model.net.state_dict(), model_path)
print(f"Model saved to {model_path}")

####################################
# try to load the model and test it
print("Loading the model...")
# Create a new network with the same architecture
loaded_net = tt.practical.MLPVanilla(in_features, hidden_layers, 1, dropout=0.1)
loaded_net.to(device)
# Load the state_dict into the network
loaded_net.load_state_dict(torch.load(model_path))
# Set the neural network to evaluation mode
loaded_net.eval()  # Call eval() on the network, not on the CoxPH model
# Create a new CoxPH model with the loaded network
model_loaded = CoxPH(loaded_net, optimizer=torch.optim.Adam)

# Print sample information
print("\nSample Information:")
print(f"Number of samples: {len(x)}")
print(f"First few samples (normalized features):")
for i in range(min(5, len(x))):
    print(f"Sample {i+1}: {x[i]}, Duration: {durations[i]}, Event: {events[i]}")
print(f"Original features (first 5 samples):")
for i in range(min(5, len(x_raw))):
    print(f"Sample {i+1}: Nedbor={x_raw[i][0]}, Nedborsminutter={x_raw[i][1]}")

# Compute baseline hazards for the loaded model
print("\nComputing baseline hazards...")
model_loaded.compute_baseline_hazards((durations, events))

H0 = model.baseline_cumulative_hazards_   # pandas Series indexed by time
S0 = np.exp(-H0)                          # baseline survival = exp(–cumhaz)

plt.figure(figsize=(8,5))
plt.step(S0.index, S0, where='post')
plt.title("Baseline Survival Curve")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(os.path.join(parent_parent_dir, 'reports', 'figures', 'baseline_survival_curve.png'))
plt.show()

# Test the loaded model with the same data
print("Making predictions with loaded model...")
surv_loaded = model_loaded.predict_surv_df(x)

# Evaluate model performance using concordance index
from pycox.evaluation import EvalSurv
ev = EvalSurv(surv_loaded, durations, events, censor_surv='km')
c_index = ev.concordance_td()
print(f"Concordance index: {c_index:.4f}")

# Plot survival curves from loaded model
plt.figure(figsize=(10, 6))
for i in range(min(5, len(x))):  # Plot survival curves for the first 5 samples
    plt.step(surv_loaded.index, surv_loaded.iloc[:, i], where="post", label=f"Sample {i+1}")
plt.title("Survival Curves (Loaded Model)")
plt.xlabel("Time (hours)")
plt.ylabel("Probability of No Heavy Rain")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(parent_parent_dir, 'reports', 'figures', 'loaded_model_curves.png'))
plt.show()

# Calculate and plot cumulative hazards
cum_haz = model_loaded.predict_cumulative_hazards(x)
plt.figure(figsize=(10, 6))
for i in range(min(5, len(x))):
    plt.step(cum_haz.index, cum_haz.iloc[:, i], where="post", label=f"Sample {i+1}")
plt.title("Cumulative Hazard Functions")
plt.xlabel("Time (hours)")
plt.ylabel("Cumulative Hazard")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(parent_parent_dir, 'reports', 'figures', 'cumulative_hazards.png'))
plt.show()

print("Model evaluation complete.")
