import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set plotting style
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (8, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

# 1. Open ROOT file
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

# Load arrays including Momentum and Length_3D
reco_fields = ["nTracks", "TrackHitPos", "TrackHitEnergies", "Momentum", "Length_3D"]
reco_arr = t_reco.arrays(reco_fields)
tr_arr = t_tr.arrays(["RecoTrackPrimaryParticlePDG"])

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
print("Extracting track-level features and PDG labels...")

for event_id in range(n_events):
    n_tracks = reco_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    track_pdgs = tr_arr["RecoTrackPrimaryParticlePDG"][event_id].tolist()
    
    momentum = ak.to_numpy(reco_arr["Momentum"][event_id])
    length_3d = ak.to_numpy(reco_arr["Length_3D"][event_id])
    
    lines = []
    track_hits = []
    track_energies = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(reco_arr["TrackHitPos"][event_id][track_idx])
        energies = ak.to_numpy(reco_arr["TrackHitEnergies"][event_id][track_idx])
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
        
    # Reconstruct vertex (reco-only) using PCA solver
    if len(lines) == 1:
        p, d, resid = lines[0]
        actual_hits = track_hits[0]
        reco_vtx = actual_hits[np.argmin(actual_hits[:, 2])]
    else:
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
        abs_pdg = abs(pdg)
        
        # Classify into 5 primary classes
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
            "length_3d": length_3d[track_idx] if track_idx < len(length_3d) else length,
            "momentum": momentum[track_idx] if track_idx < len(momentum) else 0.0,
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
        
    # Rank by length
    sorted_by_len = sorted(range(len(event_tracks)), key=lambda k: event_tracks[k]["length"], reverse=True)
    for rank, idx in enumerate(sorted_by_len):
        event_tracks[idx]["length_rank"] = rank + 1
        event_tracks[idx]["is_longest"] = 1 if rank == 0 else 0
        
    # Rank by z_depth
    sorted_by_z = sorted(range(len(event_tracks)), key=lambda k: event_tracks[k]["z_depth"], reverse=True)
    for rank, idx in enumerate(sorted_by_z):
        event_tracks[idx]["z_depth_rank"] = rank + 1
        
    track_records.extend(event_tracks)

df = pd.DataFrame(track_records)
print(f"Features compiled for {len(df)} tracks.")

# Define feature columns (including momentum and length_3d)
feature_cols = [
    "length", "length_3d", "momentum", "nhits", "z_depth", "straightness", "total_energy", "dedx",
    "start_z", "end_z", "dca", "angle", "dist_to_vtx", "length_rank", "z_depth_rank", "is_longest"
]
X = df[feature_cols]
y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Balanced Random Forest Classifier for Particle Identification...")
clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Save the model
model_path = Path("particle_pid_classifier.joblib")
joblib.dump(clf, model_path)
print(f"Saved particle PID classifier model to: {model_path.resolve()}")

# Predictions
y_pred = clf.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
classes = sorted(y.unique())
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=classes, cmap="Blues", ax=ax)
plt.title("Particle Identification Confusion Matrix (Enriched)")
plt.tight_layout()
plt.savefig("pid_confusion_matrix.png")
plt.close()
print("Saved Confusion Matrix plot: pid_confusion_matrix.png")

# Feature Importances Plot
importances = clf.feature_importances_
feat_importances = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_importances.plot(kind='barh', color='skyblue', edgecolor='k')
plt.title("Particle ID Feature Importances (Enriched)")
plt.xlabel("Relative Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("pid_feature_importances.png")
plt.close()
print("Saved Feature Importances plot: pid_feature_importances.png")
