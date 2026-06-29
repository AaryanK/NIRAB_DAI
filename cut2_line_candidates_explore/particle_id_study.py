import uproot
import awkward as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Open ROOT file
f = uproot.open("../Cut2.root")
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load arrays
rc_arr = t_rc.arrays(["TrackHitPos", "TrackHitEnergies", "nTracks"])
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "RecoTrackPrimaryParticlePDG"])

def fit_3d_line(hits):
    mean = np.mean(hits, axis=0)
    centered = hits - mean
    cov = np.cov(centered, rowvar=False)
    if cov.ndim < 2:
        return mean, np.array([0.0, 0.0, 1.0]), 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    diff = centered
    proj = diff - np.outer(diff @ direction, direction)
    residuals = np.sum(proj**2, axis=1)
    mean_residual = np.mean(residuals)
    
    return mean, direction, mean_residual

def get_dca_and_angle(vtx, p, d, start_hit):
    v_to_p = vtx - p
    dca = np.linalg.norm(v_to_p - (v_to_p @ d) * d)
    
    v_to_start = start_hit - vtx
    norm = np.linalg.norm(v_to_start)
    if norm > 0:
        cos_angle = (v_to_start @ d) / norm
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
    else:
        angle = 0.0
    return dca, angle

track_records = []

for event_id in range(len(rc_arr)):
    n_tracks = rc_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    track_pdgs = tr_arr["RecoTrackPrimaryParticlePDG"][event_id].tolist()
    
    # Fit tracks and collect temporary info
    lines = []
    track_hits = []
    track_energies = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        energies = ak.to_numpy(rc_arr["TrackHitEnergies"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        actual_energies = energies[mask]
        
        if len(actual_hits) >= 2:
            p, d, resid = fit_3d_line(actual_hits)
            lines.append((p, d, resid))
            track_hits.append(actual_hits)
            track_energies.append(actual_energies)
            
    if len(lines) == 0:
        continue
        
    # Reconstruct vertex (reco-only)
    if len(lines) == 1:
        p, d, resid = lines[0]
        actual_hits = track_hits[0]
        reco_vtx = actual_hits[np.argmin(actual_hits[:, 2])]
    else:
        # Solve vertex system
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for p, d, resid in lines:
            proj = np.eye(3) - np.outer(d, d)
            A += proj
            b += proj @ p
        try:
            reco_vtx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            reco_vtx = np.mean([p for p, d, resid in lines], axis=0)
            
    # Calculate track-level features
    event_tracks = []
    for track_idx, (p, d, resid) in enumerate(lines):
        actual_hits = track_hits[track_idx]
        actual_energies = track_energies[track_idx]
        
        start_hit = actual_hits[np.argmin(actual_hits[:, 2])]
        end_hit = actual_hits[np.argmax(actual_hits[:, 2])]
        length = np.linalg.norm(end_hit - start_hit)
        nhits = len(actual_hits)
        
        dca, angle = get_dca_and_angle(reco_vtx, p, d, start_hit)
        dist_to_vtx = np.linalg.norm(start_hit - reco_vtx)
        z_depth = np.max(actual_hits[:, 2]) - np.min(actual_hits[:, 2])
        
        total_energy = np.sum(actual_energies)
        dedx = total_energy / length if length > 0 else 0.0
        
        pdg = track_pdgs[track_idx] if track_idx < len(track_pdgs) else 0
        
        # Categorize label
        abs_pdg = abs(pdg)
        if abs_pdg == 13:
            label = "muon"
        elif abs_pdg == 211:
            label = "pion"
        elif pdg == 2212:
            label = "proton"
        elif abs_pdg == 11:
            label = "electron"
        else:
            label = "other"
            
        event_tracks.append({
            "pdg": pdg,
            "label": label,
            "length": length,
            "nhits": nhits,
            "dca": dca,
            "angle": angle,
            "dist_to_vtx": dist_to_vtx,
            "z_depth": z_depth,
            "start_z": start_hit[2],
            "end_z": end_hit[2],
            "straightness": resid,
            "total_energy": total_energy,
            "dedx": dedx
        })
        
    # Sort and rank tracks by length
    sorted_by_len = sorted(range(len(event_tracks)), key=lambda k: event_tracks[k]["length"], reverse=True)
    for rank, idx in enumerate(sorted_by_len):
        event_tracks[idx]["length_rank"] = rank + 1
        event_tracks[idx]["is_longest"] = 1 if rank == 0 else 0
        
    # Sort and rank tracks by z_depth
    sorted_by_z = sorted(range(len(event_tracks)), key=lambda k: event_tracks[k]["z_depth"], reverse=True)
    for rank, idx in enumerate(sorted_by_z):
        event_tracks[idx]["z_depth_rank"] = rank + 1
        
    track_records.extend(event_tracks)

df = pd.DataFrame(track_records)
df.to_csv("reco_pid_features.csv", index=False)
print(f"Features compiled: {len(df)} tracks.")

# Define inputs and targets
feature_cols = [
    "length", "nhits", "z_depth", "straightness", "total_energy", "dedx",
    "start_z", "end_z", "dca", "angle", "dist_to_vtx", "length_rank", "z_depth_rank", "is_longest"
]
X = df[feature_cols]
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Baseline Rule-Based Cuts ---")
# Simple cut: Muons are long and penetrate deep in Z
# Let's use a simple cut: length > 1800 mm AND z_depth > 1500 mm as a muon
y_pred_baseline = np.where((X_test["length"] > 1800) & (X_test["z_depth"] > 1500), "muon", "other")
# Map true test labels to binary 'muon' vs 'other' for baseline evaluation
y_test_binary = np.where(y_test == "muon", "muon", "other")

cm_base = confusion_matrix(y_test_binary, y_pred_baseline, labels=["muon", "other"])
tn, fp, fn, tp = cm_base.ravel()
eff_base = tp / (tp + fn) if (tp + fn) > 0 else 0
pur_base = tp / (tp + fp) if (tp + fp) > 0 else 0
rej_base = tn / (tn + fp) if (tn + fp) > 0 else 0

print("Baseline Muon vs Other confusion matrix:")
print(cm_base)
print(f"Baseline Muon Efficiency: {eff_base*100:.2f}%")
print(f"Baseline Muon Purity:     {pur_base*100:.2f}%")
print(f"Baseline Hadron/Other Rejection: {rej_base*100:.2f}%")

print("\n--- Random Forest Classifier (Multi-class) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix (Random Forest):")
classes = sorted(y.unique())
print(pd.DataFrame(confusion_matrix(y_test, y_pred_rf, labels=classes), index=classes, columns=classes))

# Binary evaluation of RF for comparison
y_pred_rf_binary = np.where(y_pred_rf == "muon", "muon", "other")
cm_rf = confusion_matrix(y_test_binary, y_pred_rf_binary, labels=["muon", "other"])
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
eff_rf = tp_rf / (tp_rf + fn_rf)
pur_rf = tp_rf / (tp_rf + fp_rf)
rej_rf = tn_rf / (tn_rf + fp_rf)

print("\nRandom Forest Binary Muon vs Other performance:")
print(f"RF Muon Efficiency: {eff_rf*100:.2f}%")
print(f"RF Muon Purity:     {pur_rf*100:.2f}%")
print(f"RF Hadron/Other Rejection: {rej_rf*100:.2f}%")
