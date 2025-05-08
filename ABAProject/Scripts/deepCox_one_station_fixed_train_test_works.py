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

# Placeholder for percolation_rates_updated (since not provided)
percolation_rates_updated = {
    'FP': {'min': 0, 'max': 1e-7},
    'FT': {'min': 0, 'max': 1e-5},  # FIX: Vary perc_rate for FT
    # Add other soil types if needed
}

# 1) Load only station 05135
station = "05135"
df_original = pd.read_parquet(f"data/processed/survival_data_{station}.parquet")
df = df_original.iloc[-50000:]
print(f"Number of columns: {len(df.columns)}")

# 2) Process data for each soil type
dfs = []
raw_soils = [c.split("_")[-1] for c in df.columns if c.startswith(f"{station}_WOG_")]
print(f"Length of soils: {len(raw_soils)}")

# Limit soils to those with 10% to 50% event rate
all_soils = False
if all_soils:
    soils = raw_soils
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

# Clip TTE to avoid zeros and large values
# FIX: Apply clipping correctly and use second-highest TTE
unique_tte = df['TTE'].unique()
top5 = np.sort(unique_tte)[-5:]
clip_upper = top5[-2]  # 961 hours
print(f"Clipping TTE to lower=1, upper={clip_upper}")
df['TTE'] = df['TTE'].clip(lower=1, upper=clip_upper)

# Add rolling sums for 6h, 12h, 24h, 72h, 168h
for win in (6, 12, 24, 72, 168):
    df[f"WOG_{win}h"] = (
        df.groupby('soil_type')['WOG']
        .apply(lambda x: x.rolling(window=win, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

# Add transformed features
df['log_WOG'] = np.log1p(df['WOG'])
df['WOG_binary'] = (df['WOG'] > 0).astype(int)

# Add percolation rates
def gather_soil_types(percolation_mapping):
    soil_types = {}
    for key, value in percolation_mapping.items():
        min_val = 0.0001 if value['min'] == 0 else value['min']
        max_val = 0.9999 if value['max'] == 1 else value['max']
        soil_types[key] = max_val  # Use max for simplicity
    return soil_types

soil_perc = gather_soil_types(percolation_rates_updated)
df['perc_rate'] = df['soil_type'].map(soil_perc)

# Save processed dataframe
df.to_csv(f"data/processed/dp_survival_data_{station}.csv", index=False)

# Print data summary
print("Dataframe head\n", df.head())
print(f"Unique values of soil types: {df['soil_type'].nunique()}")
first_index_observed_one = df[df['observed'] == 1].index[0]
print(f"First index where observed == 1: {first_index_observed_one}")
print(f"First rows around first heavy rain event:\n{df.iloc[first_index_observed_one-3:first_index_observed_one+3]}")
print(f"Event rate: {df['observed'].mean():.4f}")

# Prepare inputs for DeepCox
X_df = pd.get_dummies(
    df[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'soil_type']],
    columns=['soil_type']
)
print(f"X_df Shape: {X_df.shape}")
print(f"X_df Columns: {X_df.columns.tolist()}")

# Scale features
scaler = RobustScaler()
X = scaler.fit_transform(X_df.values.astype('float32'))

# Targets
durations = df['TTE'].values.astype('float32')
events = df['observed'].astype(int).values

print(f"Shape of X: {X.shape}")
print(f"Shape of durations: {durations.shape}, Shape of events: {events.shape}")

# Split into train and validation sets
# FIX: Add train-validation split to monitor overfitting
X_train, X_val, durations_train, durations_val, events_train, events_val = train_test_split(
    X, durations, events, test_size=0.2, random_state=42, stratify=events
)

# Build and train the Deep Cox PH model
in_features = X.shape[1]
print(f"in_features {in_features}")
# FIX: Increase model capacity and add batch normalization
net = tt.practical.MLPVanilla(in_features, [128, 64, 32], 1, dropout=0.2, batch_norm=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(f"device {device}")

# FIX: Add early stopping and increase epochs
model = CoxPH(
    net,
    optimizer=torch.optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=1e-4),  # Increased lr and weight decay
    device=device
)
early_stopping = tt.callbacks.EarlyStopping(patience=10)

print("Starting training...")
model.fit(
    X_train,
    (durations_train, events_train),
    epochs=100,  # Increased epochs
    batch_size=256,
    verbose=True,
    val_data=(X_val, (durations_val, events_val)),  # Validation data
    callbacks=[early_stopping]
)

print("Training complete. Model fitting done.")
# Compute baseline hazards
model.compute_baseline_hazards()
print("Baseline hazards ready.")
print("Baseline hazards (first 10):")
print(model.baseline_hazards_.head(10))
print(f"Total baseline hazard: {model.baseline_hazards_.sum()}")
print(f"Baseline cumulative hazards shape: {model.baseline_cumulative_hazards_.shape}")

# Save model and scaler
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)
model_name = f"deepcox_{station}_wog_with_aggregates"
torch.save(net.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))
pd.to_pickle(scaler, os.path.join(save_dir, f"scaler_{model_name}.pkl"))
print(f"Model and scaler saved with features: {X_df.columns.tolist()}")

# Sample rows with high WOG or events for testing
# FIX: Sample rows with WOG > 0 or observed == 1
sampled = df[df['WOG'] > df['WOG'].quantile(0.9)].groupby('soil_type', group_keys=False).apply(
    lambda g: g[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'soil_type']].sample(1, random_state=42)
).reset_index(drop=True)
if sampled.empty:
    sampled = df.groupby('soil_type', group_keys=False).apply(
        lambda g: g[['WOG', 'WOG_6h', 'WOG_12h', 'WOG_24h', 'WOG_72h', 'WOG_168h', 'log_WOG', 'WOG_binary', 'perc_rate', 'soil_type']].sample(1, random_state=42)
    ).reset_index(drop=True)

# One-hot encode sampled data
X_ex = pd.get_dummies(sampled, columns=['soil_type']).reindex(columns=X_df.columns, fill_value=0).astype('float32')
print("X_ex shape:", X_ex.shape)

# Inspect log-risk scores
Xt = torch.from_numpy(X_ex.values).to(device)
with torch.no_grad():
    log_risks = net(Xt).cpu().numpy().flatten()
print("Log-risk scores:")
print(pd.Series(log_risks, index=sampled['soil_type']))

# Predict survival curves
surv = model.predict_surv_df(X_ex.values)
surv.columns = sampled['soil_type'].tolist()

# Plot survival curves
ax = surv.plot(drawstyle='steps-pre', figsize=(8, 5), linewidth=2)
ax.set(
    title=f"Deep-CoxPH Survival by Soil (Station {station})",
    xlabel="Time (h)",
    ylabel="Survival Probability"
)
ax.set_xlim(0, clip_upper)
ax.grid(True)
plt.legend(title="Soil", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f"reports/figures/survival_curves_per_soil_{station}.png", dpi=150)
plt.show()

# Plot failure curves
fail = 1 - surv
ax = fail.plot(drawstyle='steps-pre', figsize=(8, 5), linewidth=2)
ax.set(
    title=f"Deep-CoxPH Failure Probability by Soil (Station {station})",
    xlabel="Time (h)",
    ylabel="Failure Probability"
)
ax.set_xlim(0, clip_upper)
ax.grid(True)
plt.legend(title="Soil", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f"reports/figures/failure_curves_per_soil_{station}.png", dpi=150)
plt.show()

# Plot baseline survival curve
H0 = model.baseline_cumulative_hazards_
S0 = np.exp(-H0)
plt.step(H0.index, S0.values.flatten(), where='post')
plt.title(f"Station {station} — Baseline Survival Curve")
plt.xlabel("Time to next heavy WOG event (hours)")
plt.ylabel("Survival Probability")
plt.grid(True)
out_dir = "reports/figures"
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, f"baseline_{station}_wog.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved baseline curve to {fig_path}")
plt.close()

# Evaluate model
# FIX: Add evaluation metrics
t_1y = min(8760, clip_upper)  # 1 year or max TTE
ev = EvalSurv(surv, durations, events, censor_surv='km')
c_index = ev.concordance_td()
brier_1y = ev.brier_score(taus=[t_1y])[0]
time_grid = np.linspace(0, t_1y, 100)
ibs_1y = ev.integrated_brier_score(time_grid)
print(f"Concordance index: {c_index:.4f}")
print(f"Brier score at {t_1y} hours: {brier_1y:.4f}")
print(f"Integrated Brier Score (0–{t_1y} h): {ibs_1y:.4f}")

# Inspect survival curves
print(f"Shape of surv: {surv.shape}")
print("\nFirst 5 survival curves:")
print(surv.head())
print("\nLast 5 survival curves:")
print(surv.tail())

# Compare with linear Cox model
# FIX: Ensure linear model uses same data
linear_model = CoxPH(None, optimizer=torch.optim.Adam(lr=1e-3), device=device)
linear_model.fit(X_train, (durations_train, events_train), epochs=100, batch_size=256, verbose=True)
linear_model.compute_baseline_hazards()
surv_linear = linear_model.predict_surv_df(X_ex.values)
ax = surv_linear.plot(drawstyle='steps-pre')
plt.title(f"Survival Curves from Linear Model (Station {station})")
plt.xlabel("Time (h)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig(f"reports/figures/linear_survival_curves_{station}.png", dpi=150)
plt.show()