import uproot
import awkward as ak
import numpy as np
import pandas as pd

# Open ROOT file
f = uproot.open("../Cut2.root")
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load arrays
print("Loading data...")
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks"])
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "TrueVtxID", "RecoTrackPrimaryParticleVtxId"])

# Convert Truth columns to list of lists to remove awkward overhead in loops
print("Converting awkward arrays to python lists/numpy for speed...")
true_vtx_x_list = [v.tolist() for v in tr_arr["TrueVtxX"]]
true_vtx_y_list = [v.tolist() for v in tr_arr["TrueVtxY"]]
true_vtx_z_list = [v.tolist() for v in tr_arr["TrueVtxZ"]]
true_vtx_id_list = [v.tolist() for v in tr_arr["TrueVtxID"]]
track_vtx_ids_list = [v.tolist() for v in tr_arr["RecoTrackPrimaryParticleVtxId"]]

# Convert Reco columns
n_tracks_list = rc_arr["nTracks"].tolist()
# Keep TrackHitPos as awkward but we will extract to numpy on track-level (fast)

# Define TMS bounds
TMS_X_MIN, TMS_X_MAX = -3730.0, 3730.0
TMS_Y_MIN, TMS_Y_MAX = -3100.0, 400.0
TMS_Z_MIN, TMS_Z_MAX = 11124.0, 18544.0

def is_inside_tms(x, y, z):
    return (TMS_X_MIN <= x <= TMS_X_MAX) and \
           (TMS_Y_MIN <= y <= TMS_Y_MAX) and \
           (TMS_Z_MIN <= z <= TMS_Z_MAX)

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

print("Processing events...")
# Process all events
for event_id in range(len(rc_arr)):
    n_tracks = n_tracks_list[event_id]
    if n_tracks < 1:
        continue
        
    vtx_ids_all = true_vtx_id_list[event_id]
    track_vtx_ids = track_vtx_ids_list[event_id]
    
    # Get all true vertices inside TMS for this event
    tms_vtxs = {}
    for idx, vtx_id in enumerate(vtx_ids_all):
        tx = true_vtx_x_list[event_id][idx]
        ty = true_vtx_y_list[event_id][idx]
        tz = true_vtx_z_list[event_id][idx]
        if is_inside_tms(tx, ty, tz):
            tms_vtxs[vtx_id] = np.array([tx, ty, tz])
            
    # Extract and fit all reco tracks in the event
    lines = []
    track_starts = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        
        if len(actual_hits) >= 2:
            p, d, resid = fit_3d_line(actual_hits)
            lines.append((p, d, resid))
            start_hit = actual_hits[np.argmin(actual_hits[:, 2])]
            track_starts.append(start_hit)
            
    if len(lines) == 0:
        continue
        
    # ----------------------------------------------------
    # MODE A: Truth-Assisted Grouping
    # ----------------------------------------------------
    truth_groups = {}
    for idx, tv_id in enumerate(track_vtx_ids):
        if tv_id in tms_vtxs and idx < len(lines):
            if tv_id not in truth_groups:
                truth_groups[tv_id] = []
            truth_groups[tv_id].append(idx)
            
    for tv_id, track_indices in truth_groups.items():
        sub_lines = [lines[i] for i in track_indices]
        if len(sub_lines) == 1:
            idx = track_indices[0]
            reco_vtx = track_starts[idx]
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
    n_lines = len(lines)
    adj = np.eye(n_lines, dtype=bool)
    
    for i in range(n_lines):
        for j in range(i+1, n_lines):
            dca, pt1, pt2 = get_dca_line_to_line(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            start_dist = np.linalg.norm(track_starts[i] - track_starts[j])
            
            if dca < 150.0 and start_dist < 400.0:
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
                comp.append(curr)
                for neighbor in range(n_lines):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(comp)
            
    for comp in clusters:
        sub_lines = [lines[i] for i in comp]
        if len(sub_lines) == 1:
            idx = comp[0]
            reco_vtx = track_starts[idx]
        else:
            reco_vtx = reconstruct_vertex(sub_lines)
            
        if len(tms_vtxs) == 0:
            continue
            
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
            dr = min_dist
            
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
                "dr": dr,
                "n_tracks_in_vertex": len(comp),
                "grouping_mode": "Reco-only"
            })

df = pd.DataFrame(records)
df.to_csv("tms_vertex_validation.csv", index=False)
print(f"Results saved to tms_vertex_validation.csv. Total records: {len(df)}")

# Print Summary Statistics by Mode
for mode in ["Truth-assisted", "Reco-only"]:
    df_mode = df[df["grouping_mode"] == mode]
    print(f"\n==========================================")
    print(f"Grouping Mode: {mode} (N={len(df_mode)})")
    print(f"==========================================")
    if len(df_mode) == 0:
        print("No records found.")
        continue
        
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
