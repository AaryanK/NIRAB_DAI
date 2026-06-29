import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Open ROOT file
file_path = Path("C:/MY_CODES/NIRAB_DAI/Cut2.root")
if not file_path.exists():
    raise FileNotFoundError("Could not find Cut2.root")

print(f"Opening ROOT file: {file_path}")
f = uproot.open(file_path)
t_reco = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]
n_events = t_reco.num_entries

# Load arrays including hit energies and hit count
reco_fields = [
    "nTracks", "StartPos", "EndPos", "StartDirection",
    "Momentum", "Length", "Length_3D", "EnergyDeposit",
    "TrackHitEnergies", "nHits"
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
    except:
        return np.mean(start_positions, axis=0)

dataset = []
for event_id in range(n_events):
    nx4 = tr_arr["NeutrinoX4"][event_id].tolist()
    true_x, true_y, true_z = nx4[0]*1000.0, nx4[1]*1000.0, nx4[2]*1000.0
    if not (11124.0 <= true_z <= 18544.0): continue
    
    n_tracks = reco_arr["nTracks"][event_id]
    start_pos = ak.to_numpy(reco_arr["StartPos"][event_id])
    end_pos = ak.to_numpy(reco_arr["EndPos"][event_id])
    start_dir = ak.to_numpy(reco_arr["StartDirection"][event_id])
    momentum = ak.to_numpy(reco_arr["Momentum"][event_id])
    length = ak.to_numpy(reco_arr["Length"][event_id])
    length_3d = ak.to_numpy(reco_arr["Length_3D"][event_id])
    energy_dep = ak.to_numpy(reco_arr["EnergyDeposit"][event_id])
    nhits = ak.to_numpy(reco_arr["nHits"][event_id])
    hit_energies = reco_arr["TrackHitEnergies"][event_id]
    
    first_hits_energies = []
    for trk_idx in range(n_tracks):
        trk_hit_en = ak.to_numpy(hit_energies[trk_idx])
        act_hit_en = trk_hit_en[trk_hit_en > -1000.0]
        first_hits_energies.append(np.sum(act_hit_en[:5]) if len(act_hit_en) > 0 else 0.0)
        
    scattering_diffs = []
    for trk_idx in range(n_tracks):
        euclidean_dist = np.linalg.norm(start_pos[trk_idx] - end_pos[trk_idx])
        scattering_diffs.append(abs(length_3d[trk_idx] - euclidean_dist))
        
    x_reco, y_reco, z_reco = 0.0, 0.0, 0.0
    if n_tracks > 0:
        if n_tracks == 1:
            x_reco, y_reco, z_reco = start_pos[0][0], start_pos[0][1], start_pos[0][2]
        else:
            vtx = reconstruct_vertex_pca(start_pos, start_dir)
            x_reco, y_reco, z_reco = vtx[0], vtx[1], vtx[2]
    else:
        x_reco, y_reco, z_reco = 0.0, 0.0, 11500.0
        
    feat = {
        "x_reco": x_reco, "y_reco": y_reco, "z_reco": z_reco,
        "nTracks": n_tracks,
        "mean_len": np.mean(length) if n_tracks > 0 else 0.0,
        "max_len": np.max(length) if n_tracks > 0 else 0.0,
        "mean_mom": np.mean(momentum) if n_tracks > 0 else 0.0,
        "max_mom": np.max(momentum) if n_tracks > 0 else 0.0,
        "mean_energy": np.mean(energy_dep) if n_tracks > 0 else 0.0,
        "sum_energy": np.sum(energy_dep) if n_tracks > 0 else 0.0,
        "min_z_start": np.min(start_pos[:, 2]) if n_tracks > 0 else 11500.0,
        "mean_z_start": np.mean(start_pos[:, 2]) if n_tracks > 0 else 11500.0,
        "mean_x_start": np.mean(start_pos[:, 0]) if n_tracks > 0 else 0.0,
        "mean_y_start": np.mean(start_pos[:, 1]) if n_tracks > 0 else 0.0,
        
        "hadronic_energy_start": np.sum(first_hits_energies) if n_tracks > 0 else 0.0,
        "max_hadronic_energy_start": np.max(first_hits_energies) if n_tracks > 0 else 0.0,
        "mean_scattering": np.mean(scattering_diffs) if n_tracks > 0 else 0.0,
        "max_scattering": np.max(scattering_diffs) if n_tracks > 0 else 0.0,
        "total_hits": np.sum(nhits) if n_tracks > 0 else 0.0,
        
        "true_x": true_x, "true_y": true_y, "true_z": true_z
    }
    dataset.append(feat)

df = pd.DataFrame(dataset)

feature_cols = [col for col in df.columns if not col.startswith("true_")]
X = df[feature_cols]
y_x, y_y, y_z = df["true_x"], df["true_y"], df["true_z"]

X_train, X_test, indices_train, indices_test = train_test_split(X, df.index, test_size=0.2, random_state=42)

# Build Voting Ensembles (Increased max_iter to 1000 and enabled early_stopping to prevent warnings and ensure convergence)
def make_ensemble():
    gbdt = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.06, subsample=0.8, random_state=42)
    rf = RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=4, random_state=42)
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1200, early_stopping=True, validation_fraction=0.1, random_state=42))
    ])
    
    ensemble = VotingRegressor([
        ('gbdt', gbdt),
        ('rf', rf),
        ('mlp', mlp)
    ], weights=[2.0, 1.0, 1.0])
    
    return ensemble

print("Training optimized ensemble models with full convergence settings...")
ens_x = make_ensemble()
ens_y = make_ensemble()
ens_z = make_ensemble()

ens_x.fit(X_train, y_x.loc[indices_train])
ens_y.fit(X_train, y_y.loc[indices_train])
ens_z.fit(X_train, y_z.loc[indices_train])

sandbox_dir = Path("C:/MY_CODES/NIRAB_DAI/sandbox")
joblib.dump(ens_x, sandbox_dir / "ensemble_regressor_x.joblib")
joblib.dump(ens_y, sandbox_dir / "ensemble_regressor_y.joblib")
joblib.dump(ens_z, sandbox_dir / "ensemble_regressor_z.joblib")
print("Saved ensemble models.")

df_test = df.loc[indices_test].copy()
X_test_feats = X.loc[indices_test]

pred_x = ens_x.predict(X_test_feats)
pred_y = ens_y.predict(X_test_feats)
pred_z = ens_z.predict(X_test_feats)

dx_before = df_test["x_reco"] - df_test["true_x"]
dy_before = df_test["y_reco"] - df_test["true_y"]
dz_before = df_test["z_reco"] - df_test["true_z"]
dr_before = np.sqrt(dx_before**2 + dy_before**2 + dz_before**2)

dx_after = pred_x - df_test["true_x"]
dy_after = pred_y - df_test["true_y"]
dz_after = pred_z - df_test["true_z"]
dr_after = np.sqrt(dx_after**2 + dy_after**2 + dz_after**2)

thresh_before = np.percentile(dr_before, 95.0)
core_mask = dr_before <= thresh_before

print("\n=== RESULTS OF OPTIMIZED CONVERGED ENSEMBLE REGRESSION (TEST CORE 95%) ===")
stats_after = pd.DataFrame({"dx": dx_after[core_mask], "dy": dy_after[core_mask], "dz": dz_after[core_mask], "dr": dr_after[core_mask]}).describe()
print(stats_after)

# Save performance comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(dy_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dy_before[core_mask]):.1f} mm")
axes[0].hist(dy_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"Ensemble: $\sigma$={np.std(dy_after[core_mask]):.1f} mm")
axes[0].set_title("Y Residual (dy) Distribution (Ensemble)")
axes[0].set_xlabel("dy (mm)")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

axes[1].hist(dr_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: median={np.median(dr_before[core_mask]):.1f} mm")
axes[1].hist(dr_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"Ensemble: median={np.median(dr_after[core_mask]):.1f} mm")
axes[1].set_title("3D Distance Error (dr) (Ensemble)")
axes[1].set_xlabel("dr (mm)")
axes[1].set_ylabel("Counts")
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Optimized Ensemble Regression Performance (Fully Converged)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(sandbox_dir / "reco_vs_true_sandbox_ensemble.png")
print("Saved sandbox plot: reco_vs_true_sandbox_ensemble.png")
