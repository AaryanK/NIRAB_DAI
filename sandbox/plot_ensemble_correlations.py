import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import uproot
import awkward as ak
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set plotting style
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (7, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

# Paths
sandbox_dir = Path("C:/MY_CODES/NIRAB_DAI/sandbox")

# Load models
ens_x = joblib.load(sandbox_dir / "ensemble_regressor_x.joblib")
ens_y = joblib.load(sandbox_dir / "ensemble_regressor_y.joblib")
ens_z = joblib.load(sandbox_dir / "ensemble_regressor_z.joblib")

# Load data
file_path = Path("C:/MY_CODES/NIRAB_DAI/Cut2.root")
f = uproot.open(file_path)
t_reco = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]
n_events = t_reco.num_entries

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

df_test = df.loc[indices_test].copy()
X_test_feats = X.loc[indices_test]

pred_x = ens_x.predict(X_test_feats)
pred_y = ens_y.predict(X_test_feats)
pred_z = ens_z.predict(X_test_feats)

# Helper function to style plots
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=10)

# --- 1. X Correlation ---
fig, ax = plt.subplots()
ax.scatter(df_test["true_x"], pred_x, color='green', alpha=0.6, edgecolors='k')
ax.plot([df_test["true_x"].min(), df_test["true_x"].max()], 
        [df_test["true_x"].min(), df_test["true_x"].max()], 'r--', lw=2, label="Ideal (y=x)")
style_plot(ax, "X Correlation: Predicted vs. Actual (Ensemble)", "Actual X (mm)", "Predicted X (mm)")
ax.legend()
plt.tight_layout()
plt.savefig(sandbox_dir / "x_correlation_ensemble.png")
plt.close()

# --- 2. Y Correlation ---
fig, ax = plt.subplots()
ax.scatter(df_test["true_y"], pred_y, color='green', alpha=0.6, edgecolors='k')
ax.plot([df_test["true_y"].min(), df_test["true_y"].max()], 
        [df_test["true_y"].min(), df_test["true_y"].max()], 'r--', lw=2, label="Ideal (y=x)")
style_plot(ax, "Y Correlation: Predicted vs. Actual (Ensemble)", "Actual Y (mm)", "Predicted Y (mm)")
ax.legend()
plt.tight_layout()
plt.savefig(sandbox_dir / "y_correlation_ensemble.png")
plt.close()

# --- 3. Z Correlation ---
fig, ax = plt.subplots()
ax.scatter(df_test["true_z"], pred_z, color='green', alpha=0.6, edgecolors='k')
ax.plot([df_test["true_z"].min(), df_test["true_z"].max()], 
        [df_test["true_z"].min(), df_test["true_z"].max()], 'r--', lw=2, label="Ideal (y=x)")
style_plot(ax, "Z Correlation: Predicted vs. Actual (Ensemble)", "Actual Z (mm)", "Predicted Z (mm)")
ax.legend()
plt.tight_layout()
plt.savefig(sandbox_dir / "z_correlation_ensemble.png")
plt.close()

print("Successfully saved predicted vs actual correlation plots for optimized ensemble")
