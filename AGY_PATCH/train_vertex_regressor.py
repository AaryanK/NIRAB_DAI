import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

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
t_lc = f["Line_Candidates;2"]
t_tr = f["Truth_Info;2"]

n_events = min(1000, t_lc.num_entries)
print(f"Loading {n_events} events for Event-Level Direct Regression training...")

lc_fields = [
    "nLinesU", "nLinesV", 
    "SlopeU", "InterceptU", "SlopeV", "InterceptV",
    "FirstHoughHitU", "LastHoughHitU", "FirstHoughHitV", "LastHoughHitV",
    "nHitsInTrackU", "nHitsInTrackV",
    "TrackLengthU", "TrackLengthV",
    "TotalTrackEnergyU", "TotalTrackEnergyV"
]
lc_arr = t_lc.arrays(lc_fields, entry_stop=n_events)
tr_arr = t_tr.arrays(["NeutrinoX4"], entry_stop=n_events)

theta = 3.0 * np.pi / 180.0
cos_half = np.cos(theta / 2.0)
sin_half = np.sin(theta / 2.0)

def get_line_start_end(event_id, view, idx):
    if view == "U":
        z1, u1 = lc_arr["FirstHoughHitU"][event_id][idx]
        z2, u2 = lc_arr["LastHoughHitU"][event_id][idx]
    else:
        z1, v1 = lc_arr["FirstHoughHitV"][event_id][idx]
        z2, v2 = lc_arr["LastHoughHitV"][event_id][idx]
    return min(z1, z2), max(z1, z2)

def fit_3d_line_from_2d(s_u, int_u, s_v, int_v):
    slope_x = (s_u + s_v) / (2.0 * cos_half)
    int_x = (int_u + int_v) / (2.0 * cos_half)
    slope_y = (s_u - s_v) / (2.0 * sin_half)
    int_y = (int_u - int_v) / (2.0 * sin_half)
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
    except np.linalg.LinAlgError:
        return np.mean([p for p, d in lines], axis=0)

event_level_dataset = []

for event_id in range(n_events):
    n_u = lc_arr["nLinesU"][event_id]
    n_v = lc_arr["nLinesV"][event_id]
    
    nx4 = tr_arr["NeutrinoX4"][event_id].tolist()
    true_x = nx4[0] * 1000.0
    true_y = nx4[1] * 1000.0
    true_z = nx4[2] * 1000.0
    
    # Filter for interactions inside the TMS region
    if not (11124.0 <= true_z <= 18544.0):
        continue
        
    # Extract line parameters
    slopes_u = lc_arr["SlopeU"][event_id].tolist()
    intercepts_u = lc_arr["InterceptU"][event_id].tolist()
    slopes_v = lc_arr["SlopeV"][event_id].tolist()
    intercepts_v = lc_arr["InterceptV"][event_id].tolist()
    
    len_u = lc_arr["TrackLengthU"][event_id].tolist()
    len_v = lc_arr["TrackLengthV"][event_id].tolist()
    energy_u = lc_arr["TotalTrackEnergyU"][event_id].tolist()
    energy_v = lc_arr["TotalTrackEnergyV"][event_id].tolist()
    nhits_u = lc_arr["nHitsInTrackU"][event_id].tolist()
    nhits_v = lc_arr["nHitsInTrackV"][event_id].tolist()
    
    # Track Z ranges
    z_starts_u = []
    z_ends_u = []
    for i in range(n_u):
        zs, ze = get_line_start_end(event_id, "U", i)
        z_starts_u.append(zs)
        z_ends_u.append(ze)
        
    z_starts_v = []
    z_ends_v = []
    for j in range(n_v):
        zs, ze = get_line_start_end(event_id, "V", j)
        z_starts_v.append(zs)
        z_ends_v.append(ze)
        
    # Compute baseline geometric vertex (if possible)
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
                x_start = ((slopes_u[i] + slopes_v[j]) / (2.0 * cos_half)) * z_start + ((intercepts_u[i] + intercepts_v[j]) / (2.0 * cos_half))
                y_start = ((slopes_u[i] - slopes_v[j]) / (2.0 * sin_half)) * z_start + ((intercepts_u[i] - intercepts_v[j]) / (2.0 * sin_half))
                start_pos = np.array([x_start, y_start, z_start])
                matched_tracks.append((start_pos, d))
                
        if matched_tracks:
            vtx = matched_tracks[0][0] if len(matched_tracks) == 1 else reconstruct_vertex_3d(matched_tracks)
            x_reco, y_reco, z_reco = vtx[0], vtx[1], vtx[2]
            
    # If no baseline vertex could be reconstructed, use average start Z as Z-reco
    if z_reco == 0.0:
        all_starts = z_starts_u + z_starts_v
        z_reco = np.mean(all_starts) if all_starts else 11500.0
        
    # Feature compilation (Handle empty lists with fallbacks)
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
        
        # Targets
        "true_x": true_x, "true_y": true_y, "true_z": true_z
    }
    
    event_level_dataset.append(feat)

df_event = pd.DataFrame(event_level_dataset)
print(f"Compiled event-level dataset for {len(df_event)} events.")

# Define feature columns
feature_cols = [col for col in df_event.columns if not col.startswith("true_")]
X = df_event[feature_cols]

y_x = df_event["true_x"]
y_y = df_event["true_y"]
y_z = df_event["true_z"]

# Split
X_train, X_test, indices_train, indices_test = train_test_split(X, df_event.index, test_size=0.2, random_state=42)

# Train three separate Gradient Boosting Regressors (with conservative hyperparameters to avoid overfitting)
reg_x = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42)
reg_y = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42)
reg_z = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, random_state=42)

reg_x.fit(X_train, y_x.loc[indices_train])
reg_y.fit(X_train, y_y.loc[indices_train])
reg_z.fit(X_train, y_z.loc[indices_train])

# Save models
joblib.dump(reg_x, "event_regressor_x.joblib")
joblib.dump(reg_y, "event_regressor_y.joblib")
joblib.dump(reg_z, "event_regressor_z.joblib")
print("Saved event-level regressors to event_regressor_[x/y/z].joblib")

# Evaluate on test set
df_test = df_event.loc[indices_test].copy()
X_test_feats = X.loc[indices_test]

# Predict true vertex directly
pred_x = reg_x.predict(X_test_feats)
pred_y = reg_y.predict(X_test_feats)
pred_z = reg_z.predict(X_test_feats)

# Calculate residuals before correction (using x_reco) vs. after direct regression
dx_before = df_test["x_reco"] - df_test["true_x"]
dy_before = df_test["y_reco"] - df_test["true_y"]
dz_before = df_test["z_reco"] - df_test["true_z"]
dr_before = np.sqrt(dx_before**2 + dy_before**2 + dz_before**2)

dx_after = pred_x - df_test["true_x"]
dy_after = pred_y - df_test["true_y"]
dz_after = pred_z - df_test["true_z"]
dr_after = np.sqrt(dx_after**2 + dy_after**2 + dz_after**2)

# Outlier filtering (worst 5% by dr_before) to check core improvement
thresh_before = np.percentile(dr_before, 95.0)
core_mask = dr_before <= thresh_before

print(f"\nEvaluation of Event-Level Direct Regression (Test N = {len(df_test)}):")
print("\n=== RESIDUALS BEFORE DIRECT REGRESSION (GEOMETRIC RECO) ===")
print(pd.DataFrame({
    "dx_before": dx_before[core_mask],
    "dy_before": dy_before[core_mask],
    "dz_before": dz_before[core_mask],
    "dr_before": dr_before[core_mask]
}).describe())

print("\n=== RESIDUALS AFTER DIRECT EVENT-LEVEL REGRESSION ===")
print(pd.DataFrame({
    "dx_after": dx_after[core_mask],
    "dy_after": dy_after[core_mask],
    "dz_after": dz_after[core_mask],
    "dr_after": dr_after[core_mask]
}).describe())

# Save comparison stats to text file
with open("event_level_regression_performance.txt", "w") as f_out:
    f_out.write("=== EVENT-LEVEL DIRECT REGRESSION PERFORMANCE (TEST CORE 95%) ===\n")
    f_out.write("\n--- BEFORE REGRESSION (GEOMETRIC) ---\n")
    f_out.write(pd.DataFrame({
        "dx_before": dx_before[core_mask], "dy_before": dy_before[core_mask],
        "dz_before": dz_before[core_mask], "dr_before": dr_before[core_mask]
    }).describe().to_string())
    f_out.write("\n\n--- AFTER DIRECT REGRESSION ---\n")
    f_out.write(pd.DataFrame({
        "dx_after": dx_after[core_mask], "dy_after": dy_after[core_mask],
        "dz_after": dz_after[core_mask], "dr_after": dr_after[core_mask]
    }).describe().to_string())
print("Saved stats to event_level_regression_performance.txt")

# Plot distributions before vs. after event-level direct regression
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Y coordinate residuals comparison (where the biggest improvement is expected)
axes[0].hist(dy_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: $\sigma$={np.std(dy_before[core_mask]):.1f} mm")
axes[0].hist(dy_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"After: $\sigma$={np.std(dy_after[core_mask]):.1f} mm")
axes[0].set_title("Y Residual Distribution")
axes[0].set_xlabel("dy (mm)")
axes[0].set_ylabel("Counts")
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 3D Distance error (dr) comparison
axes[1].hist(dr_before[core_mask], bins=20, color='red', alpha=0.5, edgecolor='black', label=f"Before: median={np.median(dr_before[core_mask]):.1f} mm")
axes[1].hist(dr_after[core_mask], bins=20, color='green', alpha=0.5, edgecolor='black', label=f"After: median={np.median(dr_after[core_mask]):.1f} mm")
axes[1].set_title("3D Distance Error (dr) Distribution")
axes[1].set_xlabel("dr (mm)")
axes[1].set_ylabel("Counts")
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Event-Level Direct ML Regression Performance (Using All Info from U & V)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("reco_vs_true_event_level_regression.png")
print("Saved reco_vs_true_event_level_regression.png")
