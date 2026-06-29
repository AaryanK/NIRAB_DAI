import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import joblib
from pathlib import Path

# Set plotting style
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

# 1. Open the ROOT file and load arrays
file_path = Path("../Cut2.root")
if not file_path.exists():
    file_path = Path("Cut2.root")
if not file_path.exists():
    raise FileNotFoundError(f"Could not find Cut2.root")

print(f"Opening ROOT file: {file_path}")
f = uproot.open(file_path)
t_lc = f["Line_Candidates;2"]
t_tr = f["Truth_Info;2"]

# We load 1000 events to build a solid dataset
n_events = min(1000, t_lc.num_entries)
print(f"Loading {n_events} events for feature extraction...")

lc_fields = [
    "nLinesU", "nLinesV", 
    "SlopeU", "InterceptU", "SlopeV", "InterceptV",
    "FirstHoughHitU", "LastHoughHitU", "FirstHoughHitV", "LastHoughHitV",
    "nHitsInTrackU", "nHitsInTrackV",
    "TrackLengthU", "TrackLengthV",
    "TotalTrackEnergyU", "TotalTrackEnergyV"
]
lc_arr = t_lc.arrays(lc_fields, entry_stop=n_events)
tr_arr = t_tr.arrays(["TrueVtxN", "TrueVtxX", "TrueVtxY", "TrueVtxZ", "TrueVtxID"], entry_stop=n_events)

theta = 3.0 * np.pi / 180.0
cos_half = np.cos(theta / 2.0)
sin_half = np.sin(theta / 2.0)

# Helper function to reconstruct 3D track properties from matched U and V line parameters
def fit_3d_line_from_2d(s_u, int_u, s_v, int_v):
    slope_x = (s_u + s_v) / (2.0 * cos_half)
    int_x = (int_u + int_v) / (2.0 * cos_half)
    
    slope_y = (s_u - s_v) / (2.0 * sin_half)
    int_y = (int_u - int_v) / (2.0 * sin_half)
    
    p0 = np.array([int_x, int_y, 0.0])
    d = np.array([slope_x, slope_y, 1.0])
    d_norm = np.linalg.norm(d)
    d = d / d_norm
    return p0, d

def get_line_start_end(event_id, view, idx):
    if view == "U":
        z1, u1 = lc_arr["FirstHoughHitU"][event_id][idx]
        z2, u2 = lc_arr["LastHoughHitU"][event_id][idx]
    else:
        z1, v1 = lc_arr["FirstHoughHitV"][event_id][idx]
        z2, v2 = lc_arr["LastHoughHitV"][event_id][idx]
        
    z_start = min(z1, z2)
    z_end = max(z1, z2)
    return z_start, z_end

def reconstruct_vertex_3d(lines):
    if len(lines) == 0:
        return None
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

# 2. Extract vertex candidates and build dataset
print("Processing events and extracting vertex candidate features...")
vertex_candidates = []

for event_id in range(n_events):
    n_u = lc_arr["nLinesU"][event_id]
    n_v = lc_arr["nLinesV"][event_id]
    
    if n_u == 0 or n_v == 0:
        continue
        
    slopes_u = lc_arr["SlopeU"][event_id].tolist()
    intercepts_u = lc_arr["InterceptU"][event_id].tolist()
    slopes_v = lc_arr["SlopeV"][event_id].tolist()
    intercepts_v = lc_arr["InterceptV"][event_id].tolist()
    
    nhits_u = lc_arr["nHitsInTrackU"][event_id].tolist()
    nhits_v = lc_arr["nHitsInTrackV"][event_id].tolist()
    len_u = lc_arr["TrackLengthU"][event_id].tolist()
    len_v = lc_arr["TrackLengthV"][event_id].tolist()
    energy_u = lc_arr["TotalTrackEnergyU"][event_id].tolist()
    energy_v = lc_arr["TotalTrackEnergyV"][event_id].tolist()
    
    # Match U and V lines to form 3D tracks
    matched_tracks = []
    u_used = set()
    v_used = set()
    
    # We rank candidate pairs by starting Z difference
    candidates = []
    for i in range(n_u):
        z_start_u, z_end_u = get_line_start_end(event_id, "U", i)
        for j in range(n_v):
            z_start_v, z_end_v = get_line_start_end(event_id, "V", j)
            
            z_start_diff = abs(z_start_u - z_start_v)
            z_end_diff = abs(z_end_u - z_end_v)
            
            if z_start_diff < 400.0 and z_end_diff < 600.0:
                candidates.append((z_start_diff, i, j, z_start_u, z_start_v, z_end_u, z_end_v))
                
    # Greedy matching
    candidates.sort()
    for diff, i, j, z_start_u, z_start_v, z_end_u, z_end_v in candidates:
        if i not in u_used and j not in v_used:
            u_used.add(i)
            v_used.add(j)
            
            p0, d = fit_3d_line_from_2d(slopes_u[i], intercepts_u[i], slopes_v[j], intercepts_v[j])
            
            z_start = (z_start_u + z_start_v) / 2.0
            slope_x = (slopes_u[i] + slopes_v[j]) / (2.0 * cos_half)
            int_x = (intercepts_u[i] + intercepts_v[j]) / (2.0 * cos_half)
            slope_y = (slopes_u[i] - slopes_v[j]) / (2.0 * sin_half)
            int_y = (intercepts_u[i] - intercepts_v[j]) / (2.0 * sin_half)
            
            x_start = slope_x * z_start + int_x
            y_start = slope_y * z_start + int_y
            start_pos = np.array([x_start, y_start, z_start])
            
            # Combine track features
            track_length = 0.5 * (len_u[i] + len_v[j])
            track_nhits = 0.5 * (nhits_u[i] + nhits_v[j])
            track_energy = 0.5 * (energy_u[i] + energy_v[j])
            track_dedx = track_energy / track_length if track_length > 0 else 0.0
            
            matched_tracks.append({
                "u_idx": i,
                "v_idx": j,
                "line": (start_pos, d),
                "start": start_pos,
                "direction": d,
                "length": track_length,
                "nhits": track_nhits,
                "energy": track_energy,
                "dedx": track_dedx
            })
            
    if not matched_tracks:
        continue
        
    # Cluster matched 3D tracks by starting positions to find vertices
    n_tracks = len(matched_tracks)
    adj = np.eye(n_tracks, dtype=bool)
    for i in range(n_tracks):
        for j in range(i+1, n_tracks):
            dist = np.linalg.norm(matched_tracks[i]["start"] - matched_tracks[j]["start"])
            if dist < 400.0:
                adj[i, j] = True
                adj[j, i] = True
                
    # Connected components
    visited = np.zeros(n_tracks, dtype=bool)
    clusters = []
    for i in range(n_tracks):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in range(n_tracks):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(comp)
            
    # Reconstruct vertex and build features for each cluster
    for comp in clusters:
        multiplicity = len(comp)
        
        # 1. Reconstruct vertex candidate position
        if multiplicity == 1:
            reco_vtx = matched_tracks[comp[0]]["start"]
        else:
            lines = [matched_tracks[idx]["line"] for idx in comp]
            reco_vtx = reconstruct_vertex_3d(lines)
            
        # 2. Extract vertex candidate features
        associated_tracks = [matched_tracks[idx] for idx in comp]
        
        # Mean distance from track starts to reconstructed vertex
        start_dists = [np.linalg.norm(reco_vtx - t["start"]) for t in associated_tracks]
        mean_start_dist = np.mean(start_dists)
        
        # DCA features of tracks to vertex
        dcas = []
        for t in associated_tracks:
            p_t, d_t = t["line"]
            v_to_p = reco_vtx - p_t
            dca_val = np.linalg.norm(v_to_p - (v_to_p @ d_t) * d_t)
            dcas.append(dca_val)
        mean_dca = np.mean(dcas)
        max_dca = np.max(dcas)
        
        # Kinematics features
        avg_len = np.mean([t["length"] for t in associated_tracks])
        avg_nhits = np.mean([t["nhits"] for t in associated_tracks])
        avg_energy = np.mean([t["energy"] for t in associated_tracks])
        avg_dedx = np.mean([t["dedx"] for t in associated_tracks])
        
        # Angular features (opening angles)
        max_opening_angle = 0.0
        mean_opening_angle = 0.0
        if multiplicity >= 2:
            angles = []
            for i_idx in range(multiplicity):
                d1 = associated_tracks[i_idx]["direction"]
                for j_idx in range(i_idx + 1, multiplicity):
                    d2 = associated_tracks[j_idx]["direction"]
                    cos_ang = np.clip(d1 @ d2, -1.0, 1.0)
                    ang = np.arccos(cos_ang) * 180.0 / np.pi
                    angles.append(ang)
            max_opening_angle = np.max(angles)
            mean_opening_angle = np.mean(angles)
            
        # 3. Match candidate with true TMS vertices for label
        true_n = tr_arr["TrueVtxN"][event_id]
        true_x = tr_arr["TrueVtxX"][event_id].tolist()
        true_y = tr_arr["TrueVtxY"][event_id].tolist()
        true_z = tr_arr["TrueVtxZ"][event_id].tolist()
        
        tms_true_vtxs = []
        for idx in range(true_n):
            tx, ty, tz = true_x[idx], true_y[idx], true_z[idx]
            if 11124.0 <= tz <= 18544.0:
                tms_true_vtxs.append(np.array([tx, ty, tz]))
                
        # Label is 1 if reconstructed vertex is within 300 mm of any true TMS vertex, 0 otherwise
        min_dist = float('inf')
        for tv in tms_true_vtxs:
            dist = np.linalg.norm(reco_vtx - tv)
            if dist < min_dist:
                min_dist = dist
                
        label = 1 if min_dist <= 300.0 else 0
        
        vertex_candidates.append({
            "event_id": event_id,
            "x_reco": reco_vtx[0],
            "y_reco": reco_vtx[1],
            "z_reco": reco_vtx[2],
            "multiplicity": multiplicity,
            "mean_start_dist": mean_start_dist,
            "mean_dca": mean_dca,
            "max_dca": max_dca,
            "avg_length": avg_len,
            "avg_nhits": avg_nhits,
            "avg_energy": avg_energy,
            "avg_dedx": avg_dedx,
            "max_opening_angle": max_opening_angle,
            "mean_opening_angle": mean_opening_angle,
            "min_dist_to_true": min_dist if min_dist != float('inf') else np.nan,
            "label": label
        })

df = pd.DataFrame(vertex_candidates)
print(f"Extracted features for {len(df)} reconstructed vertex candidates.")
print(f"Labels distribution: {df['label'].value_counts().to_dict()}")

# Save raw dataset
df.to_csv("vertex_candidate_features.csv", index=False)
print("Saved features to vertex_candidate_features.csv")

# 3. Train Classifier
print("\n--- Training Classifier ---")
# Drop non-feature columns
features_list = [
    "x_reco", "y_reco", "z_reco", "multiplicity", "mean_start_dist",
    "mean_dca", "max_dca", "avg_length", "avg_nhits", "avg_energy", "avg_dedx",
    "max_opening_angle", "mean_opening_angle"
]

# Check for nan values
df = df.dropna(subset=["label"])
X = df[features_list].fillna(0.0)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {len(X_train)} (pos={np.sum(y_train)}), Test size: {len(X_test)} (pos={np.sum(y_test)})")

# Train Random Forest Classifier with balanced class weight due to imbalance
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Train Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

# 4. Evaluation
print("\n=== Random Forest Performance ===")
y_pred_rf = rf.predict(X_test)
y_probs_rf = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
rf_auc = roc_auc_score(y_test, y_probs_rf)
print(f"ROC-AUC Score: {rf_auc:.4f}")

print("\n=== Gradient Boosting Performance ===")
y_pred_gb = gb.predict(X_test)
y_probs_gb = gb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_gb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))
gb_auc = roc_auc_score(y_test, y_probs_gb)
print(f"ROC-AUC Score: {gb_auc:.4f}")

# Plotting curves
plt.figure(figsize=(12, 5))

# ROC curve
plt.subplot(1, 2, 1)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_probs_gb)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})', color='blue')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {gb_auc:.3f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Precision-Recall curve
plt.subplot(1, 2, 2)
p_rf, r_rf, _ = precision_recall_curve(y_test, y_probs_rf)
p_gb, r_gb, _ = precision_recall_curve(y_test, y_probs_gb)
plt.plot(r_rf, p_rf, label=f'Random Forest (PR-AUC = {auc(r_rf, p_rf):.3f})', color='blue')
plt.plot(r_gb, p_gb, label=f'Gradient Boosting (PR-AUC = {auc(r_gb, p_gb):.3f})', color='orange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("vertex_classifier_curves.png")
print("Saved evaluation curves to vertex_classifier_curves.png")

# Save Feature Importances
plt.figure(figsize=(8, 5))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Random Forest Feature Importances for Vertex Identification")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='steelblue')
plt.xticks(range(X.shape[1]), [features_list[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("vertex_feature_importances.png")
print("Saved feature importances plot to vertex_feature_importances.png")

# Save the trained models
joblib.dump(rf, "vertex_rf_classifier.joblib")
joblib.dump(gb, "vertex_gb_classifier.joblib")
print("Models saved successfully.")
