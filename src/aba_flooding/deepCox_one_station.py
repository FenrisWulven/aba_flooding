import pandas as pd
import numpy as np
import torch
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from datetime import datetime
import os

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from aba_flooding.perculation_mapping import percolation_rates_updated
# from aba_flooding.model import SurivalDeepCoxModel

# 1) Load only station 05005
station = "05005"
df_original = pd.read_parquet(f"data/processed/survival_data_{station}.parquet")
# Only take the most recent 20,000 rows to speed up training time (roughly 3 years of data)
df = df_original.iloc[-50000:]
print(f"Number of columns: {len(df.columns)}")


# 2) Process data for each soil type
dfs = []
raw_soils = [c.split("_")[-1] for c in df.columns if c.startswith(f"{station}_WOG_")]
print(f"Length of soils: {len(raw_soils)}")

# limit the soils so they minimum have 10% of observed == 1 and max 50%
all_soils = True

if all_soils:
    soils = raw_soils
else:
    soils = [s for s in raw_soils
            if df[f"{station}_{s}_observed"].sum() > 1
            and 0.1 <= df[f"{station}_{s}_observed"].mean() < 0.5]
# this is only about 3 out of 20 soil types

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
# row order is already hourly‐chronological

# avoid feeding TTE = 0 by clipping so all 0 TTE values are set to 1, as there is 1 hour until the next event
df['TTE'] = df['TTE'].clip(lower=1)

#  one way to fold 6 h, 12 h and 24 h aggregates (and percolation rates) 
# Group by soil and roll over the last N rows (i.e. last N hours),
for win in (6, 12, 24):
    df[f"WOG_{win}h"] = (df
                         .groupby('soil_type')['WOG'] 
                         .apply(lambda x: x.rolling(window=win, min_periods=1).sum())
                         .reset_index(level=0, drop=True)
    )

# add percolation rates
def gather_soil_types(percolation_mapping):
    # Take perculation Keys and the min and max / 2 and add to a dict
    soil_types = {}
    for key, value in percolation_mapping.items():
        min = 0.0001 if value['min'] == 0 else value['min']
        max = 0.9999 if value['max'] == 1 else value['max']
            
        soil_types[key] = max #(min + max) / 2 # TODO: CHANGE
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
print(f"First rows of the dataframe starting from the first heavy rain event: \n{df.iloc[first_index_observed_one-10:first_index_observed_one+3]}")
# print the head of dataframe starting from first_index_observed_one
# print(f"print the head of dataframe starting from first_index_observed_one: \n{df.head(df.index[first_index_observed_one])}")



# Example if i dont remove any soil types
# Number of columns: 80
# Length of soils: 20
#    WOG     TTE  observed soil_type
# 0  0.0  219077         0        YS
# 1  0.0  219077         0        YS
# 2  0.0  219077         0        YS
# 3  0.0  219077         0        YS
# 4  0.0  219077         0        YS
# Unique values of soil types: 20


# Prepare inputs for DeepCox
# Features: just WOG and soil types
print(df.columns.tolist())

# one hot encode the soil types
X_df = pd.get_dummies(df[['WOG',  'WOG_6h', 'WOG_12h', 'WOG_24h', 'perc_rate', 'soil_type']], columns=['soil_type'])
print(f"   * X_df Shape of X: {X_df.shape}")
print(f"   * X_df Columns of X: {X_df.columns.tolist()}")
X = X_df.values.astype('float32')
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Targets: durations and events
durations = df['TTE'].values.astype('float32')
events    = df['observed'].astype(int).values

# print the shape
print(f"Shape of X: {X.shape}")
print(f"Columns of X: {X_df.columns.tolist()}")
print(f"Shape of durations: {durations.shape}, Shape of events: {events.shape}")

# Build and train the Deep Cox PH model
# Network: 2×32 MLP, dropout 0.1
in_features = X.shape[1]
print(f"in_features {in_features}")
net = tt.practical.MLPVanilla(in_features, [32, 32], 1, dropout=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(f"device {device}")

model = CoxPH(net, optimizer=torch.optim.Adam, device=device)

print("Starting training on station 05005 with WOG only…")
model.fit(X, (durations, events), epochs=20, batch_size=8700, verbose=True)

print("Training complete. Model fitting done.")
# Compute baseline hazards
model.compute_baseline_hazards()
print("Baseline hazards ready.")
print(" Baseline hazards:")
print(model.baseline_cumulative_hazards_)
# shape
print(model.baseline_cumulative_hazards_.shape)


# SAVE
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

model_name = "deepcox_05005_wog_with_aggregates"
torch.save(net.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))
pd.to_pickle(scaler, os.path.join(save_dir, f"scaler_{model_name}.pkl"))

print("Training complete. Model and scaler saved with WOG, WOG_6h, WOG_12h, WOG_24h, and perc_rate features.")

### TESTING
# medians_old = df.groupby('soil_type')['WOG'].median()
# med = df.groupby('soil_type')['WOG'].quantile(0.9)

# medians of non-zero wog
# medians_nz = (df[df.WOG > 0].groupby('soil_type')['WOG'].median())
# print("NEW median non-zero WOG\n", medians_nz)

# examples = []
# for soil, w in medians_nz.items():
#     row = {"WOG": w}
#     # this loop builds the exact same dummy columns
#     for col in X_df.columns.drop("WOG"):
#         row[col] = int(col == f"soil_type_{soil}")
#     examples.append(row)
# X_ex = scaler.transform(pd.DataFrame(examples)[X_df.columns]).astype('float32')
# surv = model.predict_surv_df(X_ex)
# surv.columns = medians_nz.index  # soil labels

# ax = surv.plot(drawstyle="steps-post", linewidth=1)
# ax.set(title="Deep-CoxPH Survival by Soil (non-zero WOG)",
#        xlabel="Time (h)", ylabel="S(t)")
# ax.grid(); 
# plt.legend(title="Soil", bbox_to_anchor=(1,1))
# plt.tight_layout()
# plt.savefig("reports/figures/survival_curves_per_soil.png")
# plt.show()

sampled = (
    df
    .groupby('soil_type', group_keys=False)
    .apply(lambda g: g.sample(1, random_state=42))
    .reset_index(drop=True)
)[['WOG','WOG_6h','WOG_12h','WOG_24h', 'perc_rate', 'soil_type']]



# one-hot encode soils & align to your training columns
X_ex = pd.get_dummies(sampled, columns=['soil_type'])
X_ex = (
    X_ex
    .reindex(columns=X_df.columns, fill_value=0)
    .astype('float32')
)

# now X_ex.shape == (n_soils, n_covariates)
print("X_ex shape:", X_ex.shape)  # e.g. (20, 25)

# inspect log-risk scores
Xt = torch.from_numpy(X_ex.values).to(device)
with torch.no_grad():
    log_risks = net(Xt).cpu().numpy().flatten()
print(pd.Series(log_risks, index=sampled['soil_type']))

# —— predict only on these 20 rows ——
surv = model.predict_surv_df(X_ex.values)
# label each column by its soil
surv.columns = sampled['soil_type'].tolist()

############## PLOT ###############33
ax = surv.plot(drawstyle='steps-pre', figsize=(8,5), linewidth=2)
ax.set(
    title="Deep-CoxPH Survival by Soil",
    xlabel="Time (h)", ylabel="Survival probability"
)
ax.set_xlim(0, int(df['TTE'].max()))
ax.grid(True)
plt.legend(title="Soil", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("reports/figures/survival_curves_per_soil_perc_rate.png")
plt.show()


fail = 1 - surv
ax = fail.plot(drawstyle='steps-pre')
ax.set(title="Failure Probability by Soil", 
       xlabel='Time (h)', ylabel='Failure Probability')
ax.grid();
ax.set_xlim(0, int(df['TTE'].max()))
plt.legend(title="Soil", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("reports/figures/failure_curves_per_soil_perc_rate.png")
plt.show()

# Plot and save the baseline survival curve
H0 = model.baseline_cumulative_hazards_
S0 = ( -H0 ).applymap(torch.exp) if hasattr(H0, 'applymap') else ( -H0 ).pipe(lambda x: np.exp(-x))
plt.step(H0.index, S0.values.flatten(), where='post')
plt.title("Station 05005 — Baseline Survival Curve (WOG only)")
plt.xlabel("Time to next heavy WOG event (hours)")
plt.ylabel("Survival Probability")
plt.grid(True)

out_dir = "reports/figures"
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, "baseline_05005_wog_only.png")
plt.savefig(fig_path, dpi=150)
print(f"Saved baseline curve to {fig_path}")
plt.close()

# Inspect a few individual survival curves
print(f"Shape of surv {surv.shape}")  
print("\nFirst survival curves (station 05005):")
print(surv.head())
time_points = surv.index


# Inspect a few individual survival curves

# # 1 year in hours
# t_1y = 8760  

# # Initialize evaluator
# ev = EvalSurv(surv, durations, events, censor_surv='km')

# # Concordance index (time-dependent C)
# c_index = ev.concordance_td()

# # Brier score at exactly 1 year
# brier_1y = ev.brier_score(taus=[t_1y])[0]

# # Integrated Brier Score from 0 to 1 year
# time_grid = np.linspace(0, t_1y, 100)
# ibs_1y = ev.integrated_brier_score(time_grid)

# print(f"Concordance index:            {c_index:.4f}")
# print(f"Brier score at 1 year:         {brier_1y:.4f}")
# print(f"Integrated Brier Score (0–1 y): {ibs_1y:.4f}")




