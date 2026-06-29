import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Set plotting style
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (15, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

# 1. Open the ROOT file
file_path = Path("../Cut2.root")
if not file_path.exists():
    file_path = Path("Cut2.root")
if not file_path.exists():
    raise FileNotFoundError("Could not find Cut2.root")

print(f"Opening ROOT file: {file_path}")
f = uproot.open(file_path)
t_reco = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

n_events = t_reco.num_entries
print(f"Loading {n_events} events from Reco_Tree for training...")

# Load required arrays
reco_fields = [
    "nTracks", "StartPos", "EndPos", "StartDirection", "EndDirection",
    "Momentum", "Length", "Length_3D", "EnergyDeposit"
]
reco_arr = t_reco.arrays(reco_fields)
tr_arr = t_tr.arrays(["NeutrinoX4"])

def reconstruct_vertex_pca(start_positions, directions):
    if len(start_positions) == 1:
        return start_positions[0]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(start_positions, directions):
        d_norm = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else d
        proj = np.eye(3) - np.outer(d_norm, d_norm)
        A += proj
        b += proj @ p
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.mean(start_positions, axis=0)

dataset = []
print("Compiling dataset from Reco_Tree...")
for event_id in range(n_events):
    nx4 = tr_arr["NeutrinoX4"][event_id].tolist()
    true_x = nx4[0] * 1000.0
    true_y = nx4[1] * 1000.0
    true_z = nx4[2] * 1000.0
    
    # Filter for interactions inside the TMS region
    if not (11124.0 <= true_z <= 18544.0):
        continue
        
    n_tracks = reco_arr["nTracks"][event_id]
    
    # Extract track properties
    start_pos = ak.to_numpy(reco_arr["StartPos"][event_id])
    end_pos = ak.to_numpy(reco_arr["EndPos"][event_id])
    start_dir = ak.to_numpy(reco_arr["StartDirection"][event_id])
    momentum = ak.to_numpy(reco_arr["Momentum"][event_id])
    length = ak.to_numpy(reco_arr["Length"][event_id])
    energy_dep = ak.to_numpy(reco_arr["EnergyDeposit"][event_id])
    
    # Compute baseline geometric vertex from 3D RecoTracks
    x_reco, y_reco, z_reco = 0.0, 0.0, 0.0
    if n_tracks > 0:
        if n_tracks == 1:
            x_reco, y_reco, z_reco = start_pos[0][0], start_pos[0][1], start_pos[0][2]
        else:
            # Multi-track PCA solver in 3D
            vtx = reconstruct_vertex_pca(start_pos, start_dir)
            x_reco, y_reco, z_reco = vtx[0], vtx[1], vtx[2]
    else:
        # Fallback to the front of the detector
        x_reco, y_reco, z_reco = 0.0, 0.0, 11500.0
        
    # Feature compilation (Handle empty lists with fallbacks)
    feat = {
        "x_reco": x_reco, "y_reco": y_reco, "z_reco": z_reco,
        "nTracks": n_tracks,
        "mean_len": np.mean(length) if n_tracks > 0 else 0.0,
        "max_len": np.max(length) if n_tracks > 0 else 0.0,
        "sum_len": np.sum(length) if n_tracks > 0 else 0.0,
        "mean_mom": np.mean(momentum) if n_tracks > 0 else 0.0,
        "max_mom": np.max(momentum) if n_tracks > 0 else 0.0,
        "sum_mom": np.sum(momentum) if n_tracks > 0 else 0.0,
        "mean_energy": np.mean(energy_dep) if n_tracks > 0 else 0.0,
        "sum_energy": np.sum(energy_dep) if n_tracks > 0 else 0.0,
        "min_z_start": np.min(start_pos[:, 2]) if n_tracks > 0 else 11500.0,
        "max_z_start": np.max(start_pos[:, 2]) if n_tracks > 0 else 18500.0,
        "mean_z_start": np.mean(start_pos[:, 2]) if n_tracks > 0 else 11500.0,
        "mean_x_start": np.mean(start_pos[:, 0]) if n_tracks > 0 else 0.0,
        "mean_y_start": np.mean(start_pos[:, 1]) if n_tracks > 0 else 0.0,
        
        # Targets
        "true_x": true_x, "true_y": true_y, "true_z": true_z
    }
    dataset.append(feat)

df_reco = pd.DataFrame(dataset)
print(f"Compiled Reco_Tree dataset with {len(df_reco)} events.")

# Train/Test split
feature_cols = [col for col in df_reco.columns if not col.startswith("true_")]
X = df_reco[feature_cols]
y_x = df_reco["true_x"]
y_y = df_reco["true_y"]
y_z = df_reco["true_z"]

X_train, X_test, indices_train, indices_test = train_test_split(X, df_reco.index, test_size=0.2, random_state=42)

# Train regressors
reg_x = GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42)
reg_y = GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42)
reg_z = GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42)

reg_x.fit(X_train, y_x.loc[indices_train])
reg_y.fit(X_train, y_y.loc[indices_train])
reg_z.fit(X_train, y_z.loc[indices_train])

# Save the models
joblib.dump(reg_x, "reco_tree_regressor_x.joblib")
joblib.dump(reg_y, "reco_tree_regressor_y.joblib")
joblib.dump(reg_z, "reco_tree_regressor_z.joblib")
print("Saved models to reco_tree_regressor_[x/y/z].joblib")

# Evaluate
df_test = df_reco.loc[indices_test].copy()
X_test_feats = X.loc[indices_test]

pred_x = reg_x.predict(X_test_feats)
pred_y = reg_y.predict(X_test_feats)
pred_z = reg_z.predict(X_test_feats)

dx_before = df_test["x_reco"] - df_test["true_x"]
dy_before = df_test["y_reco"] - df_test["true_y"]
dz_before = df_test["z_reco"] - df_test["true_z"]
dr_before = np.sqrt(dx_before**2 + dy_before**2 + dz_before**2)

dx_after = pred_x - df_test["true_x"]
dy_after = pred_y - df_test["true_y"]
dz_after = pred_z - df_test["true_z"]
dr_after = np.sqrt(dx_after**2 + dy_after**2 + dz_after**2)

# Outlier filtering (worst 5%)
thresh_before = np.percentile(dr_before, 95.0)
core_mask = dr_before <= thresh_before

print("\nEvaluation of Reco_Tree Direct Regression (Test N = 198):")
print("\n=== RESIDUALS BEFORE ML REGRESSION (GEOMETRIC PCA ON RECO_TREE) ===")
print(pd.DataFrame({
    "dx_before": dx_before[core_mask], "dy_before": dy_before[core_mask],
    "dz_before": dz_before[core_mask], "dr_before": dr_before[core_mask]
}).describe())

print("\n=== RESIDUALS AFTER ML REGRESSION ON RECO_TREE ===")
print(pd.DataFrame({
    "dx_after": dx_after[core_mask], "dy_after": dy_after[core_mask],
    "dz_after": dz_after[core_mask], "dr_after": dr_after[core_mask]
}).describe())

# Plot residual comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Y residual comparison
axes[0].hist(dy_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dy_before[core_mask]):.1f} mm")
axes[0].hist(dy_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"After: $\sigma$={np.std(dy_after[core_mask]):.1f} mm")
axes[0].set_title("Y Residual (dy) Distribution (Reco_Tree)")
axes[0].set_xlabel("dy (mm)")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 3D Distance error (dr) comparison
axes[1].hist(dr_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: median={np.median(dr_before[core_mask]):.1f} mm")
axes[1].hist(dr_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"After: median={np.median(dr_after[core_mask]):.1f} mm")
axes[1].set_title("3D Distance Error (dr) (Reco_Tree)")
axes[1].set_xlabel("dr (mm)")
axes[1].set_ylabel("Counts")
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Reco_Tree Direct ML Regression Performance (Using Fully Reconstructed 3D Tracks)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("reco_vs_true_reco_tree_regression.png")
print("Saved reco_vs_true_reco_tree_regression.png")

# Calculate containment accuracy
thresholds = [300.0, 500.0, 1000.0, 1500.0]
print("\nReco_Tree Vertex Reconstruction Accuracy:")
for th in thresholds:
    acc_before = np.mean(dr_before <= th) * 100.0
    acc_after = np.mean(dr_after <= th) * 100.0
    print(f"dr < {int(th)}mm | Before: {acc_before:.1f}% | After ML: {acc_after:.1f}% | +{acc_after-acc_before:.1f}%")
