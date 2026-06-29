import uproot
import awkward as ak
import numpy as np

# Open ROOT file
f = uproot.open("../Cut2.root")
t_rc = f["Reco_Tree;2"]
t_tr = f["Truth_Info;2"]

# Load arrays
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks"], entry_stop=20)
tr_arr = t_tr.arrays(["TrueVtxX", "TrueVtxY", "TrueVtxZ", "TrueVtxID", "RecoTrackPrimaryParticleVtxId"], entry_stop=20)

dcas = []

for event_id in range(20):
    n_tracks = rc_arr["nTracks"][event_id]
    if n_tracks < 1:
        continue
        
    vtx_ids = tr_arr["TrueVtxID"][event_id].tolist()
    track_vtx_ids = tr_arr["RecoTrackPrimaryParticleVtxId"][event_id].tolist()
    
    for track_idx in range(min(n_tracks, len(track_vtx_ids))):
        hits = ak.to_numpy(rc_arr["TrackHitPos"][event_id][track_idx])
        mask = hits[:, 0] > -9e8
        actual_hits = hits[mask]
        
        if len(actual_hits) >= 2:
            # Fit line
            mean = np.mean(actual_hits, axis=0)
            centered = actual_hits - mean
            cov = np.cov(centered, rowvar=False)
            if cov.ndim < 2:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Find true vertex index
            tv_id = track_vtx_ids[track_idx]
            if tv_id in vtx_ids:
                idx = vtx_ids.index(tv_id)
                true_vtx = np.array([
                    tr_arr["TrueVtxX"][event_id][idx],
                    tr_arr["TrueVtxY"][event_id][idx],
                    tr_arr["TrueVtxZ"][event_id][idx]
                ])
                
                # DCA to true vertex
                v_to_p = true_vtx - mean
                dca = np.linalg.norm(v_to_p - (v_to_p @ direction) * direction)
                dcas.append(dca)

print(f"Computed DCA to true vertex for {len(dcas)} tracks.")
print(f"Mean DCA to true vertex: {np.mean(dcas):.2f} mm")
print(f"Median DCA to true vertex: {np.median(dcas):.2f} mm")
print(f"Max DCA to true vertex: {np.max(dcas):.2f} mm")
