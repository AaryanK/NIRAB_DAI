import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Open ROOT file
f = uproot.open("../Cut2.root")
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load arrays
print("Loading data...")
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks", "StartPos"])
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "TrueVtxID", "RecoTrackPrimaryParticleVtxId"])

# Convert Truth columns to list of lists to remove awkward overhead in loops
print("Converting awkward arrays to python lists...")
true_vtx_x = [v.tolist() for v in tr_arr["TrueVtxX"]]
true_vtx_y = [v.tolist() for v in tr_arr["TrueVtxY"]]
true_vtx_z = [v.tolist() for v in tr_arr["TrueVtxZ"]]
true_vtx_id = [v.tolist() for v in tr_arr["TrueVtxID"]]
track_vtx_ids_list = [v.tolist() for v in tr_arr["RecoTrackPrimaryParticleVtxId"]]
n_tracks_list = rc_arr["nTracks"].tolist()

# Define TMS bounds in mm
TMS_X_MIN, TMS_X_MAX = -3730.0, 3730.0
TMS_Y_MIN, TMS_Y_MAX = -3100.0, 400.0
TMS_Z_MIN, TMS_Z_MAX = 11124.0, 18544.0

def is_inside_tms(x, y, z):
    return (TMS_X_MIN <= x <= TMS_X_MAX) and \
           (TMS_Y_MIN <= y <= TMS_Y_MAX) and \
           (TMS_Z_MIN <= z <= TMS_Z_MAX)

def fit_3d_line(hits):
    """Fit a 3D line using PCA. Returns mean, direction oriented downstream, and mean residual."""
    mean = np.mean(hits, axis=0)
    centered = hits - mean
    cov = np.cov(centered, rowvar=False)
    if cov.ndim < 2:
        return mean, np.array([0.0, 0.0, 1.0]), 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    if direction[2] < 0:
        direction = -direction
    
    diff = centered
    proj = diff - np.outer(diff @ direction, direction)
    residuals = np.sum(proj**2, axis=1)
    mean_residual = np.mean(residuals)
    return mean, direction, mean_residual

def get_dca_line_to_line(p1, d1, p2, d2):
    n = np.cross(d1, d2)
    n_len = np.linalg.norm(n)
    if n_len < 1e-6:
        return np.linalg.norm(np.cross(p2 - p1, d1)), p1, p2
    
    dca = np.abs((p2 - p1) @ n) / n_len
    
    a = d1 @ d1
    b = d1 @ d2
    c = d2 @ d2
    d = (p2 - p1) @ d1
    e = (p2 - p1) @ d2
    
    denom = a * c - b * b
    if np.abs(denom) < 1e-6:
        return dca, p1, p2
        
    s = (c * d - b * e) / denom
    t = (b * d - a * e) / denom
    
    pt1 = p1 + s * d1
    pt2 = p2 + t * d2
    return dca, pt1, pt2

def reconstruct_vertex(lines):
    if len(lines) == 0:
        return None
    if len(lines) == 1:
        return lines[0][0]
        
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for p, d, _ in lines:
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ p
        
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.mean([p for p, d, _ in lines], axis=0)

records = []

print("Running main vertex reconstruction and validation loop...")
for event_id in range(len(rc_arr)):
    n_tracks = n_tracks_list[event_id]
    if n_tracks < 1:
        continue
        
    vtx_ids_all = true_vtx_id[event_id]
    track_vtx_ids = track_vtx_ids_list[event_id]
    
    # Get all true vertices inside TMS for this event
    tms_vtxs = {}
    for idx, vtx_id in enumerate(vtx_ids_all):
        tx = true_vtx_x[event_id][idx]
        ty = true_vtx_y[event_id][idx]
        tz = true_vtx_z[event_id][idx]
        if is_inside_tms(tx, ty, tz):
            tms_vtxs[vtx_id] = np.array([tx, ty, tz])
            
    if len(tms_vtxs) == 0:
        # Skip events that don't have any true vertices in the TMS
        continue
        
    # Fit all reco tracks in the event
    reco_tracks = {}
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        
        if len(actual_hits) >= 2:
            p, d, resid = fit_3d_line(actual_hits)
            start_pos = np.array(rc_arr["StartPos"][event_id][track_idx])
            
            reco_tracks[track_idx] = {
                "line": (p, d, resid),
                "start": start_pos,
                "hits": actual_hits
            }
            
    if len(reco_tracks) == 0:
        continue
        
    # ----------------------------------------------------
    # MODE A: Truth-Assisted Grouping
    # ----------------------------------------------------
    truth_groups = {}
    for track_idx, track_info in reco_tracks.items():
        tv_id = track_vtx_ids[track_idx]
        if tv_id in tms_vtxs:
            if tv_id not in truth_groups:
                truth_groups[tv_id] = []
            truth_groups[tv_id].append(track_idx)
            
    for tv_id, track_indices in truth_groups.items():
        sub_lines = [reco_tracks[idx]["line"] for idx in track_indices]
        if len(sub_lines) == 1:
            reco_vtx = reco_tracks[track_indices[0]]["start"]
        else:
            reco_vtx = reconstruct_vertex(sub_lines)
            
        true_vtx = tms_vtxs[tv_id]
        dx = reco_vtx[0] - true_vtx[0]
        dy = reco_vtx[1] - true_vtx[1]
        dz = reco_vtx[2] - true_vtx[2]
        dr = np.linalg.norm(reco_vtx - true_vtx)
        
        records.append({
            "event_id": event_id,
            "true_vtx_id": tv_id,
            "reco_vtx_x": reco_vtx[0],
            "reco_vtx_y": reco_vtx[1],
            "reco_vtx_z": reco_vtx[2],
            "true_vtx_x": true_vtx[0],
            "true_vtx_y": true_vtx[1],
            "true_vtx_z": true_vtx[2],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dr": dr,
            "n_tracks_in_vertex": len(track_indices),
            "grouping_mode": "Truth-assisted"
        })
        
    # ----------------------------------------------------
    # MODE B: Reco-Only Grouping
    # ----------------------------------------------------
    track_indices = list(reco_tracks.keys())
    n_lines = len(track_indices)
    adj = np.eye(n_lines, dtype=bool)
    
    for i in range(n_lines):
        idx_i = track_indices[i]
        for j in range(i+1, n_lines):
            idx_j = track_indices[j]
            
            line_i = reco_tracks[idx_i]["line"]
            line_j = reco_tracks[idx_j]["line"]
            start_i = reco_tracks[idx_i]["start"]
            start_j = reco_tracks[idx_j]["start"]
            
            dca, pt1, pt2 = get_dca_line_to_line(line_i[0], line_i[1], line_j[0], line_j[1])
            start_dist = np.linalg.norm(start_i - start_j)
            
            # Check upstream intersection constraint
            is_upstream = True
            V_int = (pt1 + pt2) / 2.0
            dot_i = (start_i - V_int) @ line_i[1]
            dot_j = (start_j - V_int) @ line_j[1]
            if dot_i < -150.0 or dot_j < -150.0:
                is_upstream = False
                    
            if dca < 150.0 and start_dist < 400.0 and is_upstream:
                adj[i, j] = True
                adj[j, i] = True
                
    visited = np.zeros(n_lines, dtype=bool)
    clusters = []
    
    for i in range(n_lines):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(track_indices[curr])
                for neighbor in range(n_lines):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(comp)
            
    for comp in clusters:
        sub_lines = [reco_tracks[idx]["line"] for idx in comp]
        if len(sub_lines) == 1:
            reco_vtx = reco_tracks[comp[0]]["start"]
        else:
            reco_vtx = reconstruct_vertex(sub_lines)
            
        # Match to the nearest true TMS vertex in this event
        nearest_tv_id = None
        min_dist = float('inf')
        for tv_id, tv_pos in tms_vtxs.items():
            dist = np.linalg.norm(reco_vtx - tv_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_tv_id = tv_id
                
        if nearest_tv_id is not None:
            true_vtx = tms_vtxs[nearest_tv_id]
            dx = reco_vtx[0] - true_vtx[0]
            dy = reco_vtx[1] - true_vtx[1]
            dz = reco_vtx[2] - true_vtx[2]
            
            records.append({
                "event_id": event_id,
                "true_vtx_id": nearest_tv_id,
                "reco_vtx_x": reco_vtx[0],
                "reco_vtx_y": reco_vtx[1],
                "reco_vtx_z": reco_vtx[2],
                "true_vtx_x": true_vtx[0],
                "true_vtx_y": true_vtx[1],
                "true_vtx_z": true_vtx[2],
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "dr": min_dist,
                "n_tracks_in_vertex": len(comp),
                "grouping_mode": "Reco-only"
            })

df = pd.DataFrame(records)
df.to_csv("tms_vertex_validation.csv", index=False)
print(f"Results saved to tms_vertex_validation.csv. Total records: {len(df)}")

# Paths for plots
art_dir = Path("C:/Users/Moktan/.gemini/antigravity-cli/brain/e555bb60-df60-4c92-be1d-c8de2c650b91")
art_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# 1. Reco-vs-True Scatter Plots for X, Y, Z (Reco-only mode)
# ----------------------------------------------------
df_ro = df[df["grouping_mode"] == "Reco-only"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
coords = [("true_vtx_x", "reco_vtx_x", "X (mm)", "royalblue"),
          ("true_vtx_y", "reco_vtx_y", "Y (mm)", "forestgreen"),
          ("true_vtx_z", "reco_vtx_z", "Z (mm)", "darkorange")]

for idx, (true_c, reco_c, label, color) in enumerate(coords):
    ax = axes[idx]
    ax.scatter(df_ro[true_c], df_ro[reco_c], alpha=0.3, color=color, s=8)
    
    # 1:1 Identity line
    min_val = min(df_ro[true_c].min(), df_ro[reco_c].min())
    max_val = max(df_ro[true_c].max(), df_ro[reco_c].max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='red', alpha=0.8, linewidth=1.5)
    
    ax.set_title(f"Reco vs. True {label}")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Reco {label}")
    ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("reco_vs_true_scatter.png", dpi=150)
plt.savefig(art_dir / "reco_vs_true_scatter.png", dpi=150)
plt.close()
print("Saved reco_vs_true_scatter.png success")

# ----------------------------------------------------
# 2. 3D Plot of Reconstructed and True Vertices (Reco-only mode)
# ----------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sample 100 random vertices for clean visual separation
sample_df = df_ro.sample(min(120, len(df_ro)), random_state=42)

ax.scatter(sample_df["true_vtx_z"], sample_df["true_vtx_x"], sample_df["true_vtx_y"], 
           color='black', marker='o', s=25, label='True Vertices', depthshade=False)
ax.scatter(sample_df["reco_vtx_z"], sample_df["reco_vtx_x"], sample_df["reco_vtx_y"], 
           color='red', marker='x', s=30, label='Reco Vertices', depthshade=False)

# Connect matched pairs with lines
for _, row in sample_df.iterrows():
    ax.plot([row["true_vtx_z"], row["reco_vtx_z"]],
            [row["true_vtx_x"], row["reco_vtx_x"]],
            [row["true_vtx_y"], row["reco_vtx_y"]], color='gray', alpha=0.4, linewidth=1)

ax.set_xlabel('Z (mm)')
ax.set_ylabel('X (mm)')
ax.set_zlabel('Y (mm)')
ax.set_title("3D Event Display: Matched Reconstructed and True Vertices inside TMS")
ax.legend()
plt.tight_layout()
plt.savefig("vertices_3d_display.png", dpi=150)
plt.savefig(art_dir / "vertices_3d_display.png", dpi=150)
plt.close()
print("Saved vertices_3d_display.png success")

# ----------------------------------------------------
# 3. Performance vs Number of Tracks at the Vertex (Reco-only mode)
# ----------------------------------------------------
track_stats = df_ro.groupby("n_tracks_in_vertex")["dr"].agg(["count", "mean", "median"])
print("\nVertex Resolution vs. Track Multiplicity:")
print(track_stats)

plt.figure(figsize=(8, 5))
x_labels = [f"{n} Track" + ("s" if n > 1 else "") + f"\n(N={count})" for n, count in zip(track_stats.index, track_stats["count"])]
x_indices = np.arange(len(track_stats))

width = 0.35
plt.bar(x_indices - width/2, track_stats["mean"], width, label='Mean Residual dr', color='cornflowerblue', edgecolor='black')
plt.bar(x_indices + width/2, track_stats["median"], width, label='Median Residual dr', color='teal', edgecolor='black')

plt.xticks(x_indices, x_labels)
plt.xlabel("Number of Tracks in Reconstructed Vertex")
plt.ylabel("Residual distance dr (mm)")
plt.title("Vertex Reconstruction Resolution vs. Track Multiplicity (Reco-only)")
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("resolution_vs_tracks.png", dpi=150)
plt.savefig(art_dir / "resolution_vs_tracks.png", dpi=150)
plt.close()
print("Saved resolution_vs_tracks.png success")

# Print overall summary statistics for reporting
print("\nSummary Statistics Comparison:")
for mode in ["Truth-assisted", "Reco-only"]:
    df_mode = df[df["grouping_mode"] == mode]
    print(f"\n==========================================")
    print(f"Grouping Mode: {mode} (N={len(df_mode)})")
    print(f"==========================================")
    print("Mean +/- Std:")
    print(f"  dx: {np.mean(df_mode['dx']):.2f} +/- {np.std(df_mode['dx']):.2f} mm")
    print(f"  dy: {np.mean(df_mode['dy']):.2f} +/- {np.std(df_mode['dy']):.2f} mm")
    print(f"  dz: {np.mean(df_mode['dz']):.2f} +/- {np.std(df_mode['dz']):.2f} mm")
    print(f"  dr: {np.mean(df_mode['dr']):.2f} +/- {np.std(df_mode['dr']):.2f} mm")
    
    print("\nMedian Absolute Residuals:")
    print(f"  dx: {np.median(np.abs(df_mode['dx'])):.2f} mm")
    print(f"  dy: {np.median(np.abs(df_mode['dy'])):.2f} mm")
    print(f"  dz: {np.median(np.abs(df_mode['dz'])):.2f} mm")
    print(f"  dr: {np.median(df_mode['dr']):.2f} mm")
    
    print("\nFraction of vertices within bounds:")
    for dist in [50, 100, 200, 500]:
        frac = np.mean(df_mode['dr'] < dist) * 100.0
        print(f"  Within {dist:3d} mm: {frac:.2f}%")
