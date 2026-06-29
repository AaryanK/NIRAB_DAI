import uproot
import awkward as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Open ROOT file
f = uproot.open("../Cut2.root")
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load arrays
rc_arr = t_rc.arrays(["TrackHitPos", "TrackHitEnergies", "nTracks"])
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "TrueVtxID", "RecoTrackPrimaryParticleVtxId", "RecoTrackPrimaryParticlePDG"])

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
        
    vtx_ids = tr_arr["TrueVtxID"][event_id].tolist()
    track_vtx_ids = tr_arr["RecoTrackPrimaryParticleVtxId"][event_id].tolist()
    track_pdgs = tr_arr["RecoTrackPrimaryParticlePDG"][event_id].tolist()
    
    # Fit tracks
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
        reco_vtx = actual_hits[np.argmin(actual_hits[:, 2])]\
    
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
            
    # Calculate track features
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
        
        label = "muon" if abs_pdg == 13 else "non-muon"
        is_muon = 1 if abs_pdg == 13 else 0
            
        event_tracks.append({
            "event_id": event_id,
            "track_idx": track_idx,
            "true_pdg": pdg,
            "true_label": label,
            "is_muon": is_muon,
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

# Define features
feature_cols = [
    "length", "nhits", "z_depth", "straightness", "total_energy", "dedx",
    "start_z", "end_z", "dca", "angle", "dist_to_vtx", "length_rank", "z_depth_rank", "is_longest"
]
X = df[feature_cols]
y = df["is_muon"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get predictions and probabilities
probs = rf.predict_proba(X_test)[:, 1]

# Compute metrics
fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

precisions, recalls, thresholds_pr = precision_recall_curve(y_test, probs)

print(f"Random Forest AUC-ROC: {roc_auc:.4f}")

# Find Working Points (WPs)
# We want to identify the index corresponding to specific thresholds:
# 1. High-efficiency WP (~90% muon efficiency, which is recall = 0.90)
idx_he = np.argmin(np.abs(recalls - 0.90))
thresh_he = thresholds_pr[idx_he] if idx_he < len(thresholds_pr) else 0.5
eff_he = recalls[idx_he]
pur_he = precisions[idx_he]

# 2. Balanced WP (F1-score maximization)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
idx_bal = np.argmax(f1_scores)
thresh_bal = thresholds_pr[idx_bal] if idx_bal < len(thresholds_pr) else 0.5
eff_bal = recalls[idx_bal]
pur_bal = precisions[idx_bal]

# 3. High-purity WP (~90% muon purity, which is precision = 0.90)
idx_hp = np.argmin(np.abs(precisions - 0.90))
thresh_hp = thresholds_pr[idx_hp] if idx_hp < len(thresholds_pr) else 0.5
eff_hp = recalls[idx_hp]
pur_hp = precisions[idx_hp]

print("\n--- Recommended Working Points ---")
print(f"High-Efficiency WP: Threshold = {thresh_he:.3f} | Muon Efficiency = {eff_he*100:.2f}% | Muon Purity = {pur_he*100:.2f}%")
print(f"Balanced WP:        Threshold = {thresh_bal:.3f} | Muon Efficiency = {eff_bal*100:.2f}% | Muon Purity = {pur_bal*100:.2f}%")
print(f"High-Purity WP:     Threshold = {thresh_hp:.3f} | Muon Efficiency = {eff_hp*100:.2f}% | Muon Purity = {pur_hp*100:.2f}%")

# Save predictions for test set
test_indices = X_test.index
df_test = df.loc[test_indices].copy()
df_test["muon_score"] = probs
df_test["predicted_label"] = np.where(probs >= thresh_bal, "muon", "non-muon")

# Save final CSV
df_test.to_csv("binary_muon_predictions.csv", index=False)
print("\nPredictions saved to binary_muon_predictions.csv")
