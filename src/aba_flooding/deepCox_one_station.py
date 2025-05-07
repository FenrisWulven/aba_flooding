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
# from aba_flooding.perculation_mapping import percolation_rates_updated
# from aba_flooding.model import SurivalDeepCoxModel

# 1) Load only station 05005
station = "05005"
df_original = pd.read_parquet(f"data/processed/survival_data_{station}.parquet")
# Only take the most recent 20,000 rows to speed up training time (roughly 3 years of data)
df = df_original.iloc[-20000:]
print(f"Number of columns: {len(df.columns)}")


# 2) Process data for each soil type
dfs = []
raw_soils = [c.split("_")[-1] for c in df.columns if c.startswith(f"{station}_WOG_")]
print(f"Length of soils: {len(raw_soils)}")

# limit the soils so they minimum have 10% of observed == 1 and max 50%
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
# save df as csv
df.to_csv(f"data/processed/dp_survival_data_{station}.csv", index=False)
df = df.copy()

print(f"Unique values of soil types: {df['soil_type'].nunique()}")
# find the first index where observed == 1
first_index_observed_one = df[df['observed'] == 1].index[0]
print(f"First index where observed == 1: {first_index_observed_one}")
print(f"First rows of the dataframe starting from the first heavy rain event: \n{df.iloc[first_index_observed_one-1:]}")

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
X_df = pd.get_dummies(df[['WOG', 'soil_type']], columns=['soil_type'])
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
model.fit(X, (durations, events), epochs=25, batch_size=64, verbose=True)

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

torch.save(net.state_dict(), os.path.join(save_dir, "deepcox_05005_wog.pt"))
pd.to_pickle(scaler, os.path.join(save_dir, "scaler_05005_wog.pkl"))

print("Training complete. Model and scaler saved.")


### TESTING
# medians_old = df.groupby('soil_type')['WOG'].median()
med = df.groupby('soil_type')['WOG'].quantile(0.9)
print("NEW median 90 quantile\n", med)
examples = []
for soil, w in med.items():
    row = {"WOG": w}
    # this loop builds the exact same dummy columns
    for col in X_df.columns.drop("WOG"):
        row[col] = int(col == f"soil_type_{soil}")
    examples.append(row)
X_ex = scaler.transform(pd.DataFrame(examples)[X_df.columns])

surv = model.predict_surv_df(X_ex)
surv.columns = med.index  # soil labels

ax = surv.plot(drawstyle="steps-post", linewidth=1)
ax.set(title="Deep-CoxPH Survival by Soil (median WOG)",
       xlabel="Time (h)", ylabel="S(t)")
ax.grid(); 
plt.legend(title="Soil", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig("reports/figures/survival_curves_per_soil.png")
plt.show()

# #############################
# 4) Plot and save the baseline survival curve
# #############################
import matplotlib.pyplot as plt

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

# #############################
# 5) Inspect a few individual survival curves
# #############################

surv = model.predict_surv_df(X[:20])
print(f"Shape of surv {surv.shape}")  # Corrected from shape() to shape
print("\nFirst 5 survival curves (station 05005):")
print(surv.head())
time_points = surv.index

# # #############################
# # 5) Inspect a few individual survival curves
# # #############################
# import numpy as np
# from pycox.evaluation import EvalSurv

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




