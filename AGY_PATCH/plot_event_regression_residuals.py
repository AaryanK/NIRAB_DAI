import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import uproot
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

# Load the models
reg_x = joblib.load("event_regressor_x.joblib")
reg_y = joblib.load("event_regressor_y.joblib")
reg_z = joblib.load("event_regressor_z.joblib")

# Load data and reconstruct features (using phi = 3 deg)
file_path = Path("../Cut2.root")
if not file_path.exists():
    file_path = Path("Cut2.root")
f = uproot.open(file_path)
t_lc = f["Line_Candidates;2"]
t_tr = f["Truth_Info;2"]
n_events = min(1000, t_lc.num_entries)

lc_arr = t_lc.arrays([
    "nLinesU", "nLinesV", 
    "SlopeU", "InterceptU", "SlopeV", "InterceptV",
    "FirstHoughHitU", "LastHoughHitU", "FirstHoughHitV", "LastHoughHitV",
    "nHitsInTrackU", "nHitsInTrackV",
    "TrackLengthU", "TrackLengthV",
    "TotalTrackEnergyU", "TotalTrackEnergyV"
], entry_stop=n_events)
tr_arr = t_tr.arrays(["NeutrinoX4"], entry_stop=n_events)

# New stereo tilt angle: phi = 3 degrees
phi = 3.0 * np.pi / 180.0
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)

def get_line_start_end(event_id, view, idx):
    if view == "U":
        z1, u1 = lc_arr["FirstHoughHitU"][event_id][idx]
        z2, u2 = lc_arr["LastHoughHitU"][event_id][idx]
    else:
        z1, v1 = lc_arr["FirstHoughHitV"][event_id][idx]
        z2, v2 = lc_arr["LastHoughHitV"][event_id][idx]
    return min(z1, z2), max(z1, z2)

def fit_3d_line_from_2d(s_u, int_u, s_v, int_v):
    slope_x = (s_u + s_v) / (2.0 * cos_phi)
    int_x = (int_u + int_v) / (2.0 * cos_phi)
    slope_y = (s_u - s_v) / (2.0 * sin_phi)
    int_y = (int_u - int_v) / (2.0 * sin_phi)
    p0 = np.array([int_x, int_y, 0.0])
    d = np.array([slope_x, slope_y, 1.0])
    d = d / np.linalg.norm(d)
    return p0, d

def reconstruct_vertex_3d(lines):
    if len(lines) == 1:
        return lines[0][0]
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in lines:
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ p
    try:
        return np.linalg.solve(A, b)
    except:
        return np.mean([p for p, d in lines], axis=0)

dataset = []
for event_id in range(n_events):
    n_u = lc_arr["nLinesU"][event_id]
    n_v = lc_arr["nLinesV"][event_id]
    nx4 = tr_arr["NeutrinoX4"][event_id].tolist()
    true_x, true_y, true_z = nx4[0]*1000.0, nx4[1]*1000.0, nx4[2]*1000.0
    if not (11124.0 <= true_z <= 18544.0): continue
    
    slopes_u = lc_arr["SlopeU"][event_id].tolist()
    intercepts_u = lc_arr["InterceptU"][event_id].tolist()
    slopes_v = lc_arr["SlopeV"][event_id].tolist()
    intercepts_v = lc_arr["InterceptV"][event_id].tolist()
    len_u = lc_arr["TrackLengthU"][event_id].tolist()
    len_v = lc_arr["TrackLengthV"][event_id].tolist()
    energy_u = lc_arr["TotalTrackEnergyU"][event_id].tolist()
    energy_v = lc_arr["TotalTrackEnergyV"][event_id].tolist()
    
    z_starts_u = [get_line_start_end(event_id, "U", i)[0] for i in range(n_u)]
    z_ends_u = [get_line_start_end(event_id, "U", i)[1] for i in range(n_u)]
    z_starts_v = [get_line_start_end(event_id, "V", j)[0] for j in range(n_v)]
    z_ends_v = [get_line_start_end(event_id, "V", j)[1] for j in range(n_v)]
    
    x_reco, y_reco, z_reco = 0.0, 0.0, 0.0
    matched_tracks = []
    if n_u > 0 and n_v > 0:
        u_used, v_used = set(), set()
        candidates = []
        for i in range(n_u):
            for j in range(n_v):
                z_start_diff = abs(z_starts_u[i] - z_starts_v[j])
                z_end_diff = abs(z_ends_u[i] - z_ends_v[j])
                if z_start_diff < 400.0 and z_end_diff < 600.0:
                    candidates.append((z_start_diff, i, j))
        candidates.sort()
        for diff, i, j in candidates:
            if i not in u_used and j not in v_used:
                u_used.add(i)
                v_used.add(j)
                p0, d = fit_3d_line_from_2d(slopes_u[i], intercepts_u[i], slopes_v[j], intercepts_v[j])
                z_start = (z_starts_u[i] + z_starts_v[j]) / 2.0
                u_val = slopes_u[i] * z_start + intercepts_u[i]
                v_val = slopes_v[j] * z_start + intercepts_v[j]
                x_start = (u_val + v_val) / (2.0 * cos_phi)
                y_start = (u_val - v_val) / (2.0 * sin_phi)
                start_pos = np.array([x_start, y_start, z_start])
                matched_tracks.append((start_pos, d))
        if matched_tracks:
            vtx = matched_tracks[0][0] if len(matched_tracks) == 1 else reconstruct_vertex_3d(matched_tracks)
            x_reco, y_reco, z_reco = vtx[0], vtx[1], vtx[2]
            
    if z_reco == 0.0:
        all_starts = z_starts_u + z_starts_v
        z_reco = np.mean(all_starts) if all_starts else 11500.0
        
    feat = {
        "x_reco": x_reco, "y_reco": y_reco, "z_reco": z_reco,
        "nLinesU": n_u, "nLinesV": n_v,
        "mean_len_u": np.mean(len_u) if len_u else 0.0,
        "max_len_u": np.max(len_u) if len_u else 0.0,
        "sum_len_u": np.sum(len_u) if len_u else 0.0,
        "mean_len_v": np.mean(len_v) if len_v else 0.0,
        "max_len_v": np.max(len_v) if len_v else 0.0,
        "sum_len_v": np.sum(len_v) if len_v else 0.0,
        "mean_energy_u": np.mean(energy_u) if energy_u else 0.0,
        "sum_energy_u": np.sum(energy_u) if energy_u else 0.0,
        "mean_energy_v": np.mean(energy_v) if energy_v else 0.0,
        "sum_energy_v": np.sum(energy_v) if energy_v else 0.0,
        "min_z_start_u": np.min(z_starts_u) if z_starts_u else 11500.0,
        "max_z_start_u": np.max(z_starts_u) if z_starts_u else 18500.0,
        "min_z_start_v": np.min(z_starts_v) if z_starts_v else 11500.0,
        "max_z_start_v": np.max(z_starts_v) if z_starts_v else 18500.0,
        "mean_slope_u": np.mean(slopes_u) if slopes_u else 0.0,
        "mean_slope_v": np.mean(slopes_v) if slopes_v else 0.0,
        "true_x": true_x, "true_y": true_y, "true_z": true_z
    }
    dataset.append(feat)

df = pd.DataFrame(dataset)
feature_cols = [col for col in df.columns if not col.startswith("true_")]
X = df[feature_cols]
X_train, X_test, indices_train, indices_test = train_test_split(X, df.index, test_size=0.2, random_state=42)

df_test = df.loc[indices_test].copy()
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

# Filter core 95% of test set for plotting
thresh_before = np.percentile(dr_before, 95.0)
core_mask = dr_before <= thresh_before

# Helper function to style plots
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=10)

# --- 1. X Residual Distribution ---
fig, ax = plt.subplots()
ax.hist(dx_before[core_mask], bins=25, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dx_before[core_mask]):.1f} mm")
ax.hist(dx_after[core_mask], bins=25, color='green', alpha=0.5, edgecolor='black', label=f"After: $\sigma$={np.std(dx_after[core_mask]):.1f} mm")
style_plot(ax, "X Residual (dx) Distribution (phi = 3 deg)", "dx (mm)", "Counts")
ax.legend()
plt.tight_layout()
plt.savefig("x_residual_regression.png")
plt.close()

# --- 2. Y Residual Distribution ---
fig, ax = plt.subplots()
ax.hist(dy_before[core_mask], bins=25, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dy_before[core_mask]):.1f} mm")
ax.hist(dy_after[core_mask], bins=25, color='green', alpha=0.5, edgecolor='black', label=f"After: $\sigma$={np.std(dy_after[core_mask]):.1f} mm")
style_plot(ax, "Y Residual (dy) Distribution (phi = 3 deg)", "dy (mm)", "Counts")
ax.legend()
plt.tight_layout()
plt.savefig("y_residual_regression.png")
plt.close()

# --- 3. Z Residual Distribution ---
fig, ax = plt.subplots()
ax.hist(dz_before[core_mask], bins=25, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dz_before[core_mask]):.1f} mm")
ax.hist(dz_after[core_mask], bins=25, color='green', alpha=0.5, edgecolor='black', label=f"After: $\sigma$={np.std(dz_after[core_mask]):.1f} mm")
style_plot(ax, "Z Residual (dz) Distribution (phi = 3 deg)", "dz (mm)", "Counts")
ax.legend()
plt.tight_layout()
plt.savefig("z_residual_regression.png")
plt.close()

print("Successfully saved residual plots for direct regression (phi = 3 deg)")
