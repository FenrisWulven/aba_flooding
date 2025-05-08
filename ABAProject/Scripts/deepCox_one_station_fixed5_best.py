import pandas as pd
import numpy as np
import torch
import torchtuples as tt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import os

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from aba_flooding.perculation_mapping import percolation_rates_updated

# Create directory for saving distribution plots
import os
dist_dir = "reports/distributions"
os.makedirs(dist_dir, exist_ok=True)

# 1) Load only station 05005
station = "05135"
df_original = pd.read_parquet(f"data/processed/survival_data_{station}.parquet")
# Only take the most recent 20,000 rows to speed up training time (roughly 3 years of data)
df = df_original.iloc[-40000:]
print(f"Number of columns: {len(df.columns)}")

# 2) Process data for each soil type
dfs = []
raw_soils = [c.split("_")[-1] for c in df.columns if c.startswith(f"{station}_WOG_")]
print(f"Length of soils: {len(raw_soils)}")


# === NON-NATURAL OR NOT APPLICABLE ===
# BY: Town (1.00E-06 to 1.00E-03 m/s)
# IA: No Access (0.0 to 0.0 m/s)
# RA: Pit (0.0 to 0.0 m/s)
# LRA: Abandoned Pit (0.0 to 0.0 m/s)
# SØ: Freshwater (0.0 to 0.0 m/s)
# TA: Technical and Artificial Construction (0.0 to 0.0 m/s)
# X: Bed Unknown, No Information (0.0 to 0.0 m/s)
non_applicable_soils = ['BY', 'IA', 'RA', 'LRA', 'RÅ', 'LRÅ', 'SØ', 'TA', 'X']

outlier_soils = ['FL', 'GL', 'OL', 'DI'] # there avg wog is over 40mm

removing_soils = non_applicable_soils + outlier_soils
# Remove these non-applicable soils and outlier soils
raw_soils = [s for s in raw_soils if s not in removing_soils] 
print(f"Length of soils after removing non-applicable and outlier: {len(raw_soils)}")

# limit the soils so they minimum have 10% of observed == 1 and max 50%
all_soils = True

if all_soils:
    soils = raw_soils #[6:9]
else:
    soils = [s for s in raw_soils
            if df[f"{station}_{s}_observed"].sum() > 1
            and 0.1 <= df[f"{station}_{s}_observed"].mean() < 0.5]

print(f"Remaining soil count {len(soils)} and types: {soils}")
for soil in soils:
    sub = df[[
        f"{station}_WOG_{soil}",
        f"{station}_{soil}_duration",
        f"{station}_{soil}_TTE",
        f"{station}_{soil}_observed",
    ]].dropna()
    sub.columns = ["WOG", "duration", "TTE", "observed"]
    sub["soil_type"] = soil
    dfs.append(sub)
df = pd.concat(dfs, ignore_index=True)


# Calculate average WOG per soil type
avg_wog_per_soil = df.groupby('soil_type')['WOG'].mean().sort_values(ascending=False)
print("\nAverage WOG per soil type:")
print(avg_wog_per_soil)

# Create a bar plot of average WOG by soil type
plt.figure(figsize=(10, 6))
avg_wog_per_soil.plot(kind='bar')
plt.title('Average WOG by Soil Type')
plt.xlabel('Soil Type')
plt.ylabel('Average WOG')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"reports/figures/avg_wog_by_soil_{station}.png", dpi=150)
plt.show()

# FIX: Collapse wet episodes to max WOG event and add event_duration
def collapse_to_max_wog(df, soil_type):
    sub_df = df[df['soil_type'] == soil_type].copy().reset_index(drop=True)
    # Identify episode boundaries
    sub_df['episode_id'] = (sub_df['observed'] != sub_df['observed'].shift(1)).cumsum()
    # Group by episode and find the row with max WOG
    # episode_groups = sub_df[sub_df['observed'] == 1].groupby('episode_id')
    # max_wog_indices = episode_groups.apply(lambda x: x['WOG'].idxmax())
    episode_groups = sub_df[sub_df['observed'] == 1].groupby('episode_id')
    max_wog_indices = episode_groups.apply(lambda x: x['WOG'].idxmax(), include_groups=False)
    # Create new dataframe with only max WOG rows
    event_df = sub_df.loc[max_wog_indices].copy()
    event_df['observed'] = 1
    # Compute event duration (number of rows in each episode)
    episode_sizes = sub_df[sub_df['observed'] == 1].groupby('episode_id').size()
    event_df['event_duration'] = event_df['episode_id'].map(episode_sizes)
    # Merge back non-event rows, setting event_duration to 0
    non_event_df = sub_df[sub_df['observed'] == 0].copy()
    non_event_df['event_duration'] = 0
    result_df = pd.concat([event_df, non_event_df]).sort_index()
    return result_df[['WOG', 'duration', 'TTE', 'observed', 'soil_type', 'event_duration']]

# Apply collapse_to_max_wog to each soil type
df = pd.concat([collapse_to_max_wog(df, soil) for soil in soils], ignore_index=True)

# avoid feeding TTE = 0 by clipping so all 0 TTE values are set to 1
df['TTE'] = df['TTE'].clip(lower=1)
# Remove the initial TTE of len(df) to instead the max actual TTE
unique_tte = df['TTE'].unique()
top5 = np.sort(unique_tte)[-5:]
print("Highest 5 unique TTE values:", top5)
# clip to 2nd highest value
print(f"clipping to value {top5[-2]}")
df['TTE'] = df['TTE'].clip(lower=1, upper=top5[-2])

# FIX: Recalculate TTE as time to next event
def compute_tte(df, soil_type):
    sub_df = df[df['soil_type'] == soil_type].copy().reset_index(drop=True)
    event_indices = sub_df[sub_df['observed'] == 1].index
    sub_df['TTE'] = np.nan
    for i in range(len(sub_df)):
        next_event_idx = event_indices[event_indices >= i]
        if len(next_event_idx) > 0:
            next_event = next_event_idx[0]
            sub_df.loc[i, 'TTE'] = next_event - i + 1
        else:
            sub_df.loc[i, 'TTE'] = len(sub_df) - i
    return sub_df['TTE']

# Apply new TTE calculation
df['TTE'] = pd.concat([compute_tte(df, soil) for soil in soils], ignore_index=True)

# wherever observed==1, overwrite TTE with the *following* row’s TTE + 1
df.loc[df.observed==1, 'TTE'] = (
    df['TTE'].shift(-1) + 1
)

# one way to fold 6h, 12h and 24h aggregates (and percolation rates)
for win in (6, 12, 24, 72, 168):
    df[f"WOG_{win}h"] = (df
                        .groupby('soil_type')['WOG']
                        .apply(lambda x: x.rolling(window=win, min_periods=1).mean())
                        .reset_index(level=0, drop=True)
    )

df['log_WOG'] = np.log1p(df['WOG'])
df['WOG_binary'] = (df['WOG'] > 0).astype(int)

# add percolation rates
def gather_soil_types(percolation_mapping):
    soil_types = {}
    for key, value in percolation_mapping.items():
        min_val = 0.0001 if value['min'] == 0 else value['min']
        max_val = 0.9999 if value['max'] == 1 else value['max']
        soil_types[key] = max_val
    return soil_types

soil_perc = gather_soil_types(percolation_rates_updated)
df['perc_rate'] = df['soil_type'].map(soil_perc)

# save df as csv
df.to_csv(f"data/processed/dp_survival_data_{station}.csv", index=False)
df = df.copy()

print("Dataframe head\n", df.head())

print(f"Unique values of soil types: {df['soil_type'].nunique()}")
# find the first index where observed == 1
first_index_observed_one = df[df['observed'] == 1].index[0]
print(f"First index where observed == 1: {first_index_observed_one}")
print(f"First rows of the dataframe starting from the first heavy rain event: \n{df.iloc[first_index_observed_one-1:first_index_observed_one+13]}")

print(f"Event rate: {df['observed'].mean():.4f}")
tte = df['TTE']
print(f"\nEvent Distribution - Summary Statistics:")
print(f"TTE for events - min: {tte.min()}, max: {tte.max()}")
print(f"TTE for events - quartiles: {tte.quantile([0.25, 0.5, 0.75]).tolist()}")

# Create histogram of TTE values for events
plt.figure(figsize=(10, 5))
plt.hist(tte, bins=30, alpha=0.7)
plt.title('Distribution of Time-to-Event (TTE) for Observed Events')
plt.xlabel('Time to Event (hours)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(f"reports/figures/tte_distribution_for_events_{station}.png", dpi=150)
plt.show()


# Plot WOG and TTE distributions for each soil type
for soil in soils:
    # Subset data for this soil type
    soil_data = df[df['soil_type'] == soil]
    
    # WOG Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(soil_data['WOG'], bins=30, alpha=0.7)
    mean_wog = soil_data['WOG'].mean()
    plt.axvline(mean_wog, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_wog:.2f}')
    plt.title(f'WOG Distribution for Soil Type: {soil}')
    plt.xlabel('WOG Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dist_dir}/wog_dist_{station}_{soil}.png", dpi=150)
    plt.close()
    
    # TTE Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(soil_data['TTE'], bins=30, alpha=0.7)
    mean_tte = soil_data['TTE'].mean()
    plt.axvline(mean_tte, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_tte:.2f}')
    plt.title(f'Time-to-Event Distribution for Soil Type: {soil}')
    plt.xlabel('Time to Event (hours)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dist_dir}/tte_dist_{station}_{soil}.png", dpi=150)
    plt.close()

##################################################
#  Prepare inputs for DeepCox
###################################################
print(df.columns.tolist())

# one hot encode the soil types
X_df = pd.get_dummies(df[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 
                          'perc_rate', 'event_duration', 'soil_type']], columns=['soil_type'])
print(f"   * X_df Shape of X: {X_df.shape}")
print(f"   * X_df Columns of X: {X_df.columns.tolist()}")
X = X_df.values.astype('float32')
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Targets: durations and events
durations = df['TTE'].values.astype('float32')
events = df['observed'].astype(int).values

# print the shape
print(f"Shape of X: {X.shape}")
print(f"Columns of X: {X_df.columns.tolist()}")
print(f"Shape of durations: {durations.shape}, Shape of events: {events.shape}")

# Build and train the Deep Cox PH model
in_features = X.shape[1]
print(f"in_features {in_features}")
net = tt.practical.MLPVanilla(in_features, [64, 64, 32], 1, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(f"device {device}")

model = CoxPH(net, optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5), device=device)

print("Starting training on station 05005 with WOG only…")
model.fit(X, (durations, events), epochs=20, batch_size=256, verbose=True)

print("Training complete. Model fitting done.")
# Compute baseline hazards
model.compute_baseline_hazards()
print("Baseline hazards ready.")
print("Baseline hazards:")
print(model.baseline_cumulative_hazards_)
print(model.baseline_cumulative_hazards_.shape)

# SAVE
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

model_name = "deepcox_05005_wog_with_aggregates"
torch.save(net.state_dict(), os.path.join(save_dir, f"{model_name}_{station}.pt"))
pd.to_pickle(scaler, os.path.join(save_dir, f"scaler_{model_name}_{station}.pkl"))

print(f"Training complete. Model and scaler saved with features: {', '.join(X_df.columns)}")

the_cols = X_df.columns
### TESTING
sampled = (
    df
    .groupby('soil_type', group_keys=False)
    .apply(lambda g: g[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 
                       'log_WOG', 'WOG_binary', 'perc_rate', 'event_duration', 'soil_type']].sample(1, random_state=42))
    .reset_index(drop=True)
)[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'event_duration', 'soil_type']]

# one-hot encode soils & align to your training columns
X_ex = pd.get_dummies(sampled, columns=['soil_type'])
X_ex = (
    X_ex
    .reindex(columns=X_df.columns, fill_value=0)
    .astype('float32')
)
# Check for NaN or infinite values in X_ex
if X_ex.isna().any().any() or np.isinf(X_ex.values).any():
    print("Warning: X_ex contains NaN or infinite values. Cleaning data...")
    X_ex = X_ex.fillna(0)
    X_ex = X_ex.replace([np.inf, -np.inf], 0)

print("X_ex shape:", X_ex.shape)

# inspect log-risk scores
Xt = torch.from_numpy(X_ex.values).to(device)
with torch.no_grad():
    log_risks = net(Xt).cpu().numpy().flatten()
print("LOG RISK\n",pd.Series(log_risks, index=sampled['soil_type']))

# predict only on these sampled rows
surv = model.predict_surv_df(X_ex.values)
surv.columns = sampled['soil_type'].tolist()

####################### INSPECT SAMPLED
print("\nSampled Rows for Prediction:")
print(sampled[['soil_type', 'WOG', 'perc_rate', 'event_duration']].sort_values('WOG', ascending=False))

wog_quantiles = df.groupby('soil_type')['WOG'].quantile([0.5, 0.75, 0.9]).unstack()
print("\nWOG Quantiles by Soil Type:")
print(wog_quantiles)

sampled_indices = sampled.index
sampled_events = df.loc[sampled_indices, 'observed'].sum()
print(f"\nEvents in sampled rows: {sampled_events}/{len(sampled)} ({sampled_events/len(sampled):.1%})")
print(f"Overall event rate: {df['observed'].mean():.1%}")

above_median = (sampled['WOG'] > 
                sampled['soil_type'].map(df.groupby('soil_type')['WOG'].median()))
print(f"Sampled rows above median WOG for their soil type: {above_median.sum()}/{len(sampled)} ({above_median.mean():.1%})")

############## PLOT
ax = surv.plot(drawstyle='steps-pre', figsize=(8,5), linewidth=2)
ax.set(
    title="Deep-CoxPH Survival by Soil",
    xlabel="Time (h)", ylabel="Survival probability"
)
ax.set_xlim(0, int(df['TTE'].max()))
ax.grid(True)
plt.legend(title="Soil", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"reports/figures/survival_curves_per_soil_perc_rate_{station}.png")
plt.show()

fail = 1 - surv
ax = fail.plot(drawstyle='steps-pre')
ax.set(title="Failure Probability by Soil", 
       xlabel='Time (h)', ylabel='Failure Probability')
ax.grid()
ax.set_xlim(0, int(df['TTE'].max()))
plt.legend(title="Soil", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(f"reports/figures/failure_curves_per_soil_perc_rate_{station}.png")
plt.show()

# Plot and save the baseline survival curve
H0 = model.baseline_cumulative_hazards_
S0 = (-H0).applymap(torch.exp) if hasattr(H0, 'applymap') else (-H0).pipe(lambda x: np.exp(-x))
plt.step(H0.index, S0.values.flatten(), where='post')
plt.title("Station 05005 — Baseline Survival Curve (WOG only)")
plt.xlabel("Time to next heavy WOG event (hours)")
plt.ylabel("Survival Probability")
plt.grid(True)

out_dir = "reports/figures"
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, f"baseline_05005_wog_only_{station}.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved baseline curve to {fig_path}")
plt.close()

# Inspect a few individual survival curves
print(f"Shape of surv {surv.shape}")  
print("\nFirst survival curves (station 05005):")
print(surv.head())
time_points = surv.index
print(f"last points", surv.tail(-5))

