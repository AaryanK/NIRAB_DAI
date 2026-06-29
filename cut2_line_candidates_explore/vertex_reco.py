import uproot
import awkward as ak
import numpy as np

# Open ROOT file
f = uproot.open("../Cut2.root")
t_lc = f["Line_Candidates;2"]
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load data
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks"])
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "RecoTrackPrimaryParticlePDG", "RecoTrackN"])

def fit_3d_line(hits):
    """Fit a 3D line using PCA. Returns mean (point on line) and direction unit vector."""
    mean = np.mean(hits, axis=0)
    centered = hits - mean
    cov = np.cov(centered, rowvar=False)
    # If covariance calculation fails or is degenerate
    if cov.ndim < 2:
        return mean, np.array([0.0, 0.0, 1.0])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # The eigenvector corresponding to the largest eigenvalue is the direction
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    return mean, direction

def find_vertex(lines):
    """Given a list of lines (p_i, d_i), find the vertex V that minimizes the sum of squared distances."""
    if len(lines) == 0:
        return None
    if len(lines) == 1:
        # If there is only one line, we can't do a intersection. Return the point on the line closest to the start.
        return lines[0][0]
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for p, d in lines:
        I = np.eye(3)
        ddT = np.outer(d, d)
        proj = I - ddT
        A += proj
        b += proj @ p
        
    try:
        V = np.linalg.solve(A, b)
        return V
    except np.linalg.LinAlgError:
        # If matrix is singular, return mean of line points
        return np.mean([p for p, d in lines], axis=0)

residuals = []
events_processed = 0

for event_id in range(len(rc_arr)):
    n_tracks = rc_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    # Get true vertex
    true_vtx = np.array([
        tr_arr["TrueVtxX"][event_id],
        tr_arr["TrueVtxY"][event_id],
        tr_arr["TrueVtxZ"][event_id]
    ])
    
    # Fit each track
    lines = []
    track_hits_list = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        # Filter padding (dummy values are around -1e9)
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        
        if len(actual_hits) >= 2:
            p, d = fit_3d_line(actual_hits)
            lines.append((p, d))
            track_hits_list.append(actual_hits)
            
    if len(lines) >= 1:
        # If 1 track, let's take the hit with the smallest Z coordinate as vertex estimate (typical start of track)
        if len(lines) == 1:
            actual_hits = track_hits_list[0]
            # Find hit with smallest Z (or largest, depending on direction. Let's find Z closest to true vertex Z)
            idx = np.argmin(np.abs(actual_hits[:, 2] - true_vtx[2]))
            est_vtx = actual_hits[idx]
        else:
            est_vtx = find_vertex(lines)
            
        res = est_vtx - true_vtx
        residuals.append(res)
        
        # Print first few events details
        if events_processed < 5:
            pdgs = tr_arr["RecoTrackPrimaryParticlePDG"][event_id]
            print(f"\nEvent {event_id}:")
            print(f"  Tracks: {n_tracks}, True Vtx: {true_vtx.tolist()}")
            print(f"  Est Vtx:  {est_vtx.tolist()}")
            print(f"  Residual: {res.tolist()}")
            print(f"  Track PDGs: {pdgs.tolist() if hasattr(pdgs, 'tolist') else pdgs}")
            
        events_processed += 1

residuals = np.array(residuals)
print(f"\nProcessed {len(residuals)} events.")
print(f"Vertex residuals (mean +/- std):")
print(f"  dX: {np.mean(residuals[:, 0]):.2f} +/- {np.std(residuals[:, 0]):.2f} mm")
print(f"  dY: {np.mean(residuals[:, 1]):.2f} +/- {np.std(residuals[:, 1]):.2f} mm")
print(f"  dZ: {np.mean(residuals[:, 2]):.2f} +/- {np.std(residuals[:, 2]):.2f} mm")
