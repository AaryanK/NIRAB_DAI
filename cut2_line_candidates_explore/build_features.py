import uproot
import awkward as ak
import numpy as np
import pandas as pd

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

features = []

for event_id in range(len(rc_arr)):
    n_tracks = rc_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    vtx_ids = tr_arr["TrueVtxID"][event_id].tolist()
    track_vtx_ids = tr_arr["RecoTrackPrimaryParticleVtxId"][event_id].tolist()
    
    # Fit each track first to get lines
    lines = []
    track_hits_list = []
    track_energies_list = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        energies = ak.to_numpy(rc_arr["TrackHitEnergies"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        actual_energies = energies[mask]
        
        if len(actual_hits) >= 2:
            p, d, resid = fit_3d_line(actual_hits)
            lines.append((p, d, resid))
            track_hits_list.append(actual_hits)
            track_energies_list.append(actual_energies)
            
    if len(lines) == 0:
        continue
        
    # Build track features
    for track_idx, (p, d, resid) in enumerate(lines[:len(track_vtx_ids)]):
        actual_hits = track_hits_list[track_idx]
        actual_energies = track_energies_list[track_idx]
        
        # Track length (first to last hit distance)
        start_hit = actual_hits[np.argmin(actual_hits[:, 2])]
        end_hit = actual_hits[np.argmax(actual_hits[:, 2])]
        length = np.linalg.norm(end_hit - start_hit)
        
        nhits = len(actual_hits)
        
        # Get true vertex for this track
        tv_id = track_vtx_ids[track_idx]
        if tv_id in vtx_ids:
            vtx_idx = vtx_ids.index(tv_id)
            vtx = np.array([
                tr_arr["TrueVtxX"][event_id][vtx_idx],
                tr_arr["TrueVtxY"][event_id][vtx_idx],
                tr_arr["TrueVtxZ"][event_id][vtx_idx]
            ])
        else:
            # Fallback to mean of track points
            vtx = np.mean([p_val for p_val, d_val, resid_val in lines], axis=0)
            
        dca, angle = get_dca_and_angle(vtx, p, d, start_hit)
        z_depth = np.max(actual_hits[:, 2]) - np.min(actual_hits[:, 2])
        
        total_energy = np.sum(actual_energies)
        dedx = total_energy / length if length > 0 else 0.0
        
        pdg = track_vtx_ids[track_idx] # wait, RecoTrackPrimaryParticlePDG is PDG code!
        # Ah, we need RecoTrackPrimaryParticlePDG instead of RecoTrackPrimaryParticleVtxId for PDG code!
        # Let's fix that below.
        
features_data = []
# Let's re-read properly to get PDG and Vertex ID separately
pdg_arr = tr_arr["RecoTrackPrimaryParticlePDG"]

for event_id in range(len(rc_arr)):
    n_tracks = rc_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    vtx_ids = tr_arr["TrueVtxID"][event_id].tolist()
    track_vtx_ids = tr_arr["RecoTrackPrimaryParticleVtxId"][event_id].tolist()
    track_pdgs = pdg_arr[event_id].tolist()
    
    # Fit tracks
    lines = []
    track_hits_list = []
    track_energies_list = []
    
    for track_idx in range(n_tracks):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        energies = ak.to_numpy(rc_arr["TrackHitEnergies"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        actual_energies = energies[mask]
        
        if len(actual_hits) >= 2:
            p, d, resid = fit_3d_line(actual_hits)
            lines.append((p, d, resid))
            track_hits_list.append(actual_hits)
            track_energies_list.append(actual_energies)
            
    if len(lines) == 0:
        continue
        
    for track_idx, (p, d, resid) in enumerate(lines[:min(len(track_vtx_ids), len(track_pdgs))]):
        actual_hits = track_hits_list[track_idx]
        actual_energies = track_energies_list[track_idx]
        
        start_hit = actual_hits[np.argmin(actual_hits[:, 2])]
        end_hit = actual_hits[np.argmax(actual_hits[:, 2])]
        length = np.linalg.norm(end_hit - start_hit)
        nhits = len(actual_hits)
        
        tv_id = track_vtx_ids[track_idx]
        if tv_id in vtx_ids:
            vtx_idx = vtx_ids.index(tv_id)
            vtx = np.array([
                tr_arr["TrueVtxX"][event_id][vtx_idx],
                tr_arr["TrueVtxY"][event_id][vtx_idx],
                tr_arr["TrueVtxZ"][event_id][vtx_idx]
            ])
        else:
            vtx = np.mean([p_val for p_val, d_val, resid_val in lines], axis=0)
            
        dca, angle = get_dca_and_angle(vtx, p, d, start_hit)
        z_depth = np.max(actual_hits[:, 2]) - np.min(actual_hits[:, 2])
        total_energy = np.sum(actual_energies)
        dedx = total_energy / length if length > 0 else 0.0
        
        pdg = track_pdgs[track_idx]
        is_muon = 1 if abs(pdg) == 13 else 0
        
        features_data.append({
            "pdg": pdg,
            "is_muon": is_muon,
            "length": length,
            "nhits": nhits,
            "dca": dca,
            "angle": angle,
            "z_depth": z_depth,
            "straightness": resid,
            "total_energy": total_energy,
            "dedx": dedx
        })

df = pd.DataFrame(features_data)
print(f"Total reconstructed tracks processed: {len(df)}")
print("\n--- Mean values for Muons vs Hadrons/Others (True Vtx Matching) ---")
print(df.groupby("is_muon").mean().drop(columns="pdg"))
df.to_csv("track_features.csv", index=False)
print("Features saved to track_features.csv")
