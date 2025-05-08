import pandas as pd
import numpy as np
import torch
import torchtuples as tt
from sklearn.preprocessing import RobustScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from aba_flooding.perculation_mapping import percolation_rates_updated

# Create directory for saving distribution plots
dist_dir = "reports/distributions"
os.makedirs(dist_dir, exist_ok=True)

# 1) Load only station 05005
station = "05135"
df_original = pd.read_parquet(f"data/processed/survival_data_{station}.parquet")
# Only take the most recent 40,000 rows to speed up training time (roughly 3 years of data)
df = df_original.iloc[-50000:]
df = df_original.copy()
print(f"Number of columns: {len(df.columns)}")

# 2) Process data for each soil type
dfs = []
raw_soils = [c.split("_")[-1] for c in df.columns if c.startswith(f"{station}_WOG_")]
print(f"Length of soils: {len(raw_soils)}")

# Remove non-applicable and outlier soils
non_applicable_soils = ['BY', 'IA', 'RA', 'LRA', 'RÅ', 'LRÅ', 'SØ', 'TA', 'X']
outlier_soils = ['FL', 'GL', 'OL', 'DI']
removing_soils = non_applicable_soils + outlier_soils
raw_soils = [s for s in raw_soils if s not in removing_soils]
print(f"Length of soils after removing non-applicable and outlier: {len(raw_soils)}")

# Limit the soils so they minimum have 3% of observed == 1 and max 20%
all_soils = True
if all_soils:
    # index 4 and 9 specifially
    soils = raw_soils
    #soils = [raw_soils[4], raw_soils[9]]
else:
    soils = [s for s in raw_soils
             if df[f"{station}_{s}_observed"].sum() > 1
             and 0.01 <= df[f"{station}_{s}_observed"].mean() < 0.2]
print(f"Remaining soil count {len(soils)} and types: {soils}")

for soil in soils:
    sub = df[[
        f"{station}_WOG_{soil}",
        f"{station}_{soil}_duration", # we actually dont need this one as TTE is used
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
    sub_df['episode_id'] = (sub_df['observed'] != sub_df['observed'].shift(1)).cumsum()
    episode_groups = sub_df[sub_df['observed'] == 1].groupby('episode_id')
    max_wog_indices = episode_groups.apply(lambda x: x['WOG'].idxmax(), include_groups=False)
    event_df = sub_df.loc[max_wog_indices].copy()
    event_df['observed'] = 1
    episode_sizes = sub_df[sub_df['observed'] == 1].groupby('episode_id').size()
    event_df['event_duration'] = event_df['episode_id'].map(episode_sizes)
    non_event_df = sub_df[sub_df['observed'] == 0].copy()
    non_event_df['event_duration'] = 0
    result_df = pd.concat([event_df, non_event_df]).sort_index()
    return result_df[['WOG', 'duration', 'TTE', 'observed', 'soil_type', 'event_duration']]

print(f"Collapsing subsequent events to a single event with max WOG and event duration")
df = pd.concat([collapse_to_max_wog(df, soil) for soil in soils], ignore_index=True)

# Avoid feeding TTE = 0 by clipping
df['TTE'] = df['TTE'].clip(lower=1)

# Recalculate TTE as time to next event
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

print(f"Computing TTE")
df['TTE'] = pd.concat([compute_tte(df, soil) for soil in soils], ignore_index=True)

# Use duration for event rows to reflect the time at which the event occurs
# df.loc[df['observed'] == 1, 'TTE'] = df.loc[df['observed'] == 1, 'duration']

# wherever observed==1, overwrite TTE with the *following* row’s TTE + 1
df.loc[df.observed==1, 'TTE'] = (df['TTE'].shift(-1) + 1)

# Clip TTE to a reasonable upper bound (2nd highest unique value or 8760)
unique_tte = df['TTE'].unique()
top5 = np.sort(unique_tte)[-5:]
clip_upper = min(top5[-2], 8760) if len(top5) > 1 else 8760
print(f"Clipping TTE to lower=1, upper={clip_upper}")
df['TTE'] = df['TTE'].clip(lower=1, upper=clip_upper)

print(f"Computing covariates - rolling averages")
# Add rolling averages for 6h, 12h, 24h, 72h, 168h
for win in (6, 12, 24, 72, 168):
    df[f"WOG_{win}h"] = (
        df.groupby('soil_type')['WOG']
        .apply(lambda x: x.rolling(window=win, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

df['log_WOG'] = np.log1p(df['WOG'])
df['WOG_binary'] = (df['WOG'] > 0).astype(int)

# Add percolation rates
def gather_soil_types(percolation_mapping):
    soil_types = {}
    for key, value in percolation_mapping.items():
        min_val = 0.0001 if value['min'] == 0 else value['min']
        max_val = 0.9999 if value['max'] == 1 else value['max']
        soil_types[key] = max_val
    return soil_types

soil_perc = gather_soil_types(percolation_rates_updated)
df['perc_rate'] = df['soil_type'].map(soil_perc)

# Save processed dataframe
df.to_csv(f"data/processed/dp_survival_data_{station}.csv", index=False)
# load csv
# df = pd.read_csv(f"data/processed/dp_survival_data_{station}.csv")

tte = df['TTE']
# Create histogram of TTE values for events
plt.figure(figsize=(10, 5))
plt.hist(tte, bins=30, alpha=0.7)
plt.title('Distribution of Time-to-Event (TTE) for Observed Events')
plt.xlabel('Time to Event (hours)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(f"reports/figures/tte_distribution_for_events_{station}.png", dpi=150)
plt.show()

print("Dataframe head\n", df.head())
print(f"Unique values of soil types: {df['soil_type'].nunique()}")
first_index_observed_one = df[df['observed'] == 1].index[0]
print(f"First index where observed == 1: {first_index_observed_one}")
print(f"First rows around first heavy rain event:\n{df.iloc[first_index_observed_one-3:first_index_observed_one+3]}")
print(f"Event rate: {df['observed'].mean():.4f}")

# Prepare inputs for DeepCox
X_df = pd.get_dummies(
    df[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'event_duration', 'soil_type']],
    columns=['soil_type']
)
print(f"X_df Shape: {X_df.shape}")
print(f"X_df Columns: {X_df.columns.tolist()}")

# Check for NaN or infinite values
# if X_df.isna().any().any() or np.isinf(X_df.values).any():
#     print("Warning: X_df contains NaN or infinite values. Cleaning data...")
#     X_df = X_df.fillna(0)
#     X_df = X_df.replace([np.inf, -np.inf], 0)

# Scale features
scaler = RobustScaler()
X = scaler.fit_transform(X_df.values.astype('float32'))

# Targets
durations = df['TTE'].values.astype('float32')
events = df['observed'].astype(int).values

print(f"Shape of X: {X.shape}")
print(f"Shape of durations: {durations.shape}, Shape of events: {events.shape}")

# Add train-test split
X_train, X_val, durations_train, durations_val, events_train, events_val = train_test_split(
    X, durations, events, test_size=0.2, random_state=42, stratify=events
)

# Build and train the Deep Cox PH model
in_features = X.shape[1]
print(f"in_features {in_features}")
net = tt.practical.MLPVanilla(in_features, [64, 64, 32], 1, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(f"device {device}")

model = CoxPH(
    net,
    optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-5),
    device=device
)
early_stopping = tt.callbacks.EarlyStopping(patience=10)

print("Starting training on station 05005 with WOG only…")
model.fit(
    X_train,
    (durations_train, events_train),
    epochs=10,  # Increased epochs with early stopping
    batch_size=256,
    verbose=True,
    val_data=(X_val, (durations_val, events_val)),
    callbacks=[early_stopping]
)

print("Training complete. Model fitting done.")
model.compute_baseline_hazards()
print("Baseline hazards ready.")
print("Baseline hazards (first 10):")
print(model.baseline_hazards_.head(10))
print(f"Total baseline hazard: {model.baseline_hazards_.sum()}")
print(f"Baseline cumulative hazards shape: {model.baseline_cumulative_hazards_.shape}")

# Save model and scaler
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
model_name = f"deepcox_05005_wog_with_aggregates"
torch.save(net.state_dict(), os.path.join(save_dir, f"{model_name}_{station}.pt"))
pd.to_pickle(scaler, os.path.join(save_dir, f"scaler_{model_name}_{station}.pkl"))
print(f"Model and scaler saved with features: {X_df.columns.tolist()}")

# Sample rows with high WOG or events for testing
# Remove 'soil_type' from the column list since it's the grouping column
feature_cols = ['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'event_duration']
sampled = df[df['observed'] == 1].groupby('soil_type', group_keys=False).apply(
    lambda g: g[feature_cols].sample(1, random_state=42) if not g.empty else pd.DataFrame(),
    include_groups=False
).reset_index()

# Reintroduce soil_type column after sampling
if not sampled.empty:
    sampled['soil_type'] = sampled.index.map(lambda x: soils[x] if x < len(soils) else None)

# Fallback to high WOG values if some soil types have no events
missing_soils = set(soils) - set(sampled['soil_type'])
if missing_soils:
    fallback_samples = df[df['soil_type'].isin(missing_soils) & (df['WOG'] > df['WOG'].quantile(0.9))].groupby('soil_type', group_keys=False).apply(
        lambda g: g[feature_cols].sample(1, random_state=42) if not g.empty else pd.DataFrame(),
        include_groups=False
    ).reset_index()
    if not fallback_samples.empty:
        fallback_samples['soil_type'] = fallback_samples.index.map(lambda x: list(missing_soils)[x] if x < len(missing_soils) else None)
    sampled = pd.concat([sampled, fallback_samples]).reset_index(drop=True)

# Ensure all soil types are represented
if len(sampled) < len(soils):
    remaining_soils = set(soils) - set(sampled['soil_type'])
    additional_samples = df[df['soil_type'].isin(remaining_soils)].groupby('soil_type', group_keys=False).apply(
        lambda g: g[feature_cols].sample(1, random_state=42),
        include_groups=False
    ).reset_index()
    if not additional_samples.empty:
        additional_samples['soil_type'] = additional_samples.index.map(lambda x: list(remaining_soils)[x] if x < len(remaining_soils) else None)
    sampled = pd.concat([sampled, additional_samples]).reset_index(drop=True)

# One-hot encode sampled data
X_ex = pd.get_dummies(sampled, columns=['soil_type']).reindex(columns=X_df.columns, fill_value=0).astype('float32')
print("X_ex shape:", X_ex.shape)

# Check for NaN or infinite values in X_ex
# if X_ex.isna().any().any() or np.isinf(X_ex.values).any():
#     print("Warning: X_ex contains NaN or infinite values. Cleaning data...")
#     X_ex = X_ex.fillna(0)
#     X_ex = X_ex.replace([np.inf, -np.inf], 0)

# Inspect log-risk scores
Xt = torch.from_numpy(X_ex.values).to(device)
with torch.no_grad():
    log_risks = net(Xt).cpu().numpy().flatten()
print("Log-risk scores:")
print(pd.Series(log_risks, index=sampled['soil_type']))

# Predict survival curves
surv = model.predict_surv_df(X_ex.values)
surv.columns = sampled['soil_type'].tolist()

# Inspect sampled rows
print("\nSampled Rows for Prediction:")
print(sampled[['soil_type', 'WOG', 'perc_rate', 'event_duration']].sort_values('WOG', ascending=False))

wog_quantiles = df.groupby('soil_type')['WOG'].quantile([0.5, 0.75, '0.9']).unstack()
print("\nWOG Quantiles by Soil Type:")
print(wog_quantiles)

# Evaluate sampled data
sampled_indices = sampled.index
sampled_events = df.loc[sampled_indices, 'observed'].sum()
print(f"\nEvents in sampled rows: {sampled_events}/{len(sampled)} ({sampled_events/len(sampled):.1%})")
print(f"Overall event rate: {df['observed'].mean():.1%}")

above_median = (sampled['WOG'] > 
                sampled['soil_type'].map(df.groupby('soil_type')['WOG'].median()))
print(f"Sampled rows above median WOG for their soil type: {above_median.sum()}/{len(sampled)} ({above_median.mean():.1%})")

# Plot survival curves
plt.figure(figsize=(8, 5))
try:
    surv_clean = surv.replace([np.inf, -np.inf], np.nan).dropna()
    if not surv_clean.empty:
        ax = surv_clean.plot(drawstyle='steps-pre', linewidth=2)
        ax.set(
            title=f"Deep-CoxPH Survival by Soil (Station {station})",
            xlabel="Time (h)", ylabel="Survival Probability"
        )
        #ax.set_xlim(0, surv.index.max() if not surv.index.empty else clip_upper)
        ax.grid(True)
        plt.legend(title="Soil", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"reports/figures/survival_curves_per_soil_{station}.png", dpi=150)
        plt.show()
    else:
        print("Survival dataframe is empty after cleaning. Cannot plot survival curves.")
except Exception as e:
    print(f"Error plotting survival curves: {e}")

# Plot failure curves
plt.figure(figsize=(8, 5))
try:
    fail = 1 - surv_clean
    if not fail.empty:
        ax = fail.plot(drawstyle='steps-pre', linewidth=2)
        ax.set(
            title=f"Deep-CoxPH Failure Probability by Soil (Station {station})",
            xlabel="Time (h)", ylabel="Failure Probability"
        )
        #ax.set_xlim(0, surv.index.max() if not surv.index.empty else clip_upper)
        ax.grid(True)
        plt.legend(title="Soil", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"reports/figures/failure_curves_per_soil_{station}.png", dpi=150)
        plt.show()
    else:
        print("Failure dataframe is empty. Cannot plot failure curves.")
except Exception as e:
    print(f"Error plotting failure curves: {e}")

# Plot baseline survival curve
H0 = model.baseline_cumulative_hazards_
plt.figure(figsize=(8, 5))
try:
    if H0 is not None and not H0.empty:
        H0_clean = H0.replace([np.inf, -np.inf], np.nan).dropna()
        if not H0_clean.empty:
            S0 = np.exp(-H0_clean.values.flatten())
            plt.step(H0_clean.index, S0, where='post')
            plt.title(f"Station {station} — Baseline Survival Curve (WOG only)")
            plt.xlabel("Time to next heavy WOG event (hours)")
            plt.ylabel("Survival Probability")
            # plt.xlim(0, clip_upper)
            plt.grid(True)
            out_dir = "reports/figures"
            os.makedirs(out_dir, exist_ok=True)
            fig_path = os.path.join(out_dir, f"baseline_{station}_wog_only.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Saved baseline curve to {fig_path}")
            plt.show()
        else:
            print("Baseline cumulative hazards are empty after cleaning. Cannot plot baseline survival curve.")
    else:
        print("Baseline cumulative hazards are empty or invalid. Cannot plot baseline survival curve.")
except Exception as e:
    print(f"Error plotting baseline survival curve: {e}")
finally:
    plt.close()

# Evaluate model
# t_1y = min(8760, clip_upper)
# ev = EvalSurv(surv, durations, events, censor_surv='km')
# c_index = ev.concordance_td()
# brier_1y = ev.brier_score(taus=[t_1y])[0]
# time_grid = np.linspace(0, t_1y, 100)
# ibs_1y = ev.integrated_brier_score(time_grid)
# print(f"Concordance index: {c_index:.4f}")
# print(f"Brier score at {t_1y} hours: {brier_1y:.4f}")
# print(f"Integrated Brier Score (0–{t_1y} h): {ibs_1y:.4f}")

# # Inspect survival curves
# print(f"Shape of surv: {surv.shape}")
# print("\nFirst 5 survival curves:")
# print(surv.head())
# print("\nLast 5 survival curves:")
# print(surv.tail())