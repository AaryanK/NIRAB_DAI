import uproot
import awkward as ak
import numpy as np

# Open ROOT file
f = uproot.open("../Cut2.root")
t_lc = f["Line_Candidates;2"]
t_rc = f["Reco_Tree;2"]

# Load first event
lc_arr = t_lc.arrays(["TrackHitPosU", "TrackHitPosV", "SlopeU", "InterceptU"], entry_stop=5)
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks"], entry_stop=5)

event_id = 0
print(f"--- Event {event_id} ---")
print(f"Line_Candidates num tracks in event: {len(lc_arr['SlopeU'][event_id])}")
print(f"Reco_Tree num tracks in event: {rc_arr['nTracks'][event_id]}")

# Let's inspect Reco_Tree TrackHitPos for event 0
# TrackHitPos is events * tracks * hits * 3
reco_tracks_hits = rc_arr["TrackHitPos"][event_id]
for trk_idx, trk_hits in enumerate(reco_tracks_hits):
    # filter out padded hits in Reco Track (padded with -1e9 or similar)
    np_hits = ak.to_numpy(trk_hits)
    mask = (np_hits[:, 0] > -9e8) & (np_hits[:, 1] > -9e8) & (np_hits[:, 2] > -9e8)
    actual_reco = np_hits[mask]
    print(f"  Reco Track {trk_idx}: {len(actual_reco)} active hits")
    if len(actual_reco) > 0:
        print(f"    First 3 hits (x, y, z): {actual_reco[:3].tolist()}")
        print(f"    Z range: {np.min(actual_reco[:, 2]):.1f} to {np.max(actual_reco[:, 2]):.1f}")

# Now let's try to match U and V hits for Line Candidates track 0
u_hits = ak.to_numpy(lc_arr["TrackHitPosU"][event_id][0])
mask_u = (u_hits[:, 0] != 0.0) | (u_hits[:, 1] != 0.0)
u_hits = u_hits[mask_u]

v_hits = ak.to_numpy(lc_arr["TrackHitPosV"][event_id][0])
mask_v = (v_hits[:, 0] != 0.0) | (v_hits[:, 1] != 0.0)
v_hits = v_hits[mask_v]

print(f"\nLine Candidate Track 0:")
print(f"  U hits: {len(u_hits)}")
print(f"  V hits: {len(v_hits)}")

# Match by Z
matched_pairs = []
for z_u, u_val in u_hits:
    # Find nearest hit in V
    diffs = np.abs(v_hits[:, 0] - z_u)
    idx = np.argmin(diffs)
    z_v, v_val = v_hits[idx]
    # Keep if they are close in Z (e.g., within 100mm)
    if diffs[idx] < 100.0:
        matched_pairs.append((z_u, u_val, z_v, v_val))

print(f"Matched pairs count: {len(matched_pairs)}")

# Let's test stereo reconstruction with theta = 3 degrees
theta = 3 * np.pi / 180.0

# Convention A
# x = (u + v)/(2*cos(theta/2)), y = (u - v)/(2*sin(theta/2))
# Convention B
# x = u, y = (v - u*cos(theta))/sin(theta)

reco_A = []
reco_B = []
for z_u, u_val, z_v, v_val in matched_pairs:
    z_avg = (z_u + z_v) / 2.0
    
    # Convention A
    xA = (u_val + v_val) / (2.0 * np.cos(theta / 2.0))
    yA = (u_val - v_val) / (2.0 * np.sin(theta / 2.0))
    reco_A.append((xA, yA, z_avg))
    
    # Convention B
    xB = u_val
    yB = (v_val - u_val * np.cos(theta)) / np.sin(theta)
    reco_B.append((xB, yB, z_avg))

# Print first few reconstructed points and compare with Reco_Tree Track 0
print("\nFirst 3 Reconstructed points (Convention A):")
for p in reco_A[:3]:
    print(f"  x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}")

print("\nFirst 3 Reconstructed points (Convention B):")
for p in reco_B[:3]:
    print(f"  x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}")

# Let's compare with actual Reco Track 0 hits
reco_track0 = ak.to_numpy(reco_tracks_hits[0])
mask_reco0 = (reco_track0[:, 0] > -9e8)
actual_reco0 = reco_track0[mask_reco0]

# Find average differences for matched Z
diffs_A = []
diffs_B = []
for pA in reco_A:
    # Find nearest hit in actual_reco0 by Z
    z_dist = np.abs(actual_reco0[:, 2] - pA[2])
    idx = np.argmin(z_dist)
    if z_dist[idx] < 50.0:
        true_x, true_y, true_z = actual_reco0[idx]
        diffs_A.append((pA[0] - true_x, pA[1] - true_y, pA[2] - true_z))

for pB in reco_B:
    z_dist = np.abs(actual_reco0[:, 2] - pB[2])
    idx = np.argmin(z_dist)
    if z_dist[idx] < 50.0:
        true_x, true_y, true_z = actual_reco0[idx]
        diffs_B.append((pB[0] - true_x, pB[1] - true_y, pB[2] - true_z))

if diffs_A:
    diffs_A = np.array(diffs_A)
    print(f"\nConvention A residuals (mean +/- std):")
    print(f"  dx: {np.mean(diffs_A[:, 0]):.2f} +/- {np.std(diffs_A[:, 0]):.2f}")
    print(f"  dy: {np.mean(diffs_A[:, 1]):.2f} +/- {np.std(diffs_A[:, 1]):.2f}")
    print(f"  dz: {np.mean(diffs_A[:, 2]):.2f} +/- {np.std(diffs_A[:, 2]):.2f}")

if diffs_B:
    diffs_B = np.array(diffs_B)
    print(f"\nConvention B residuals (mean +/- std):")
    print(f"  dx: {np.mean(diffs_B[:, 0]):.2f} +/- {np.std(diffs_B[:, 0]):.2f}")
    print(f"  dy: {np.mean(diffs_B[:, 1]):.2f} +/- {np.std(diffs_B[:, 1]):.2f}")
    print(f"  dz: {np.mean(diffs_B[:, 2]):.2f} +/- {np.std(diffs_B[:, 2]):.2f}")
