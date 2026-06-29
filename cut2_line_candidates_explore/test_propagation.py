import uproot
import awkward as ak
import numpy as np

# Open ROOT file
f = uproot.open("../Cut2.root")
t_lc = f["Line_Candidates;2"]
t_rc = f["Reco_Tree;2"]

# Load first event
lc_arr = t_lc.arrays(["TrackHitPosU", "TrackHitPosV", "SlopeU", "InterceptU", "SlopeV", "InterceptV"], entry_stop=5)
rc_arr = t_rc.arrays(["TrackHitPos", "nTracks"], entry_stop=5)

event_id = 0
print(f"--- Propagation Test Event {event_id} ---")

u_hits = ak.to_numpy(lc_arr["TrackHitPosU"][event_id][0])
mask_u = (u_hits[:, 0] != 0.0) | (u_hits[:, 1] != 0.0)
u_hits = u_hits[mask_u]

v_hits = ak.to_numpy(lc_arr["TrackHitPosV"][event_id][0])
mask_v = (v_hits[:, 0] != 0.0) | (v_hits[:, 1] != 0.0)
v_hits = v_hits[mask_v]

slope_u = lc_arr["SlopeU"][event_id][0]
slope_v = lc_arr["SlopeV"][event_id][0]
intercept_u = lc_arr["InterceptU"][event_id][0]
intercept_v = lc_arr["InterceptV"][event_id][0]

print(f"SlopeU={slope_u:.4f}, InterceptU={intercept_u:.1f}")
print(f"SlopeV={slope_v:.4f}, InterceptV={intercept_v:.1f}")

# Match by Z
matched_pairs = []
for z_u, u_val in u_hits:
    diffs = np.abs(v_hits[:, 0] - z_u)
    idx = np.argmin(diffs)
    z_v, v_val = v_hits[idx]
    if diffs[idx] < 100.0:
        matched_pairs.append((z_u, u_val, z_v, v_val))

# Let's test stereo reconstruction with theta = 3 degrees (0.0523598 rad)
theta = 3.0 * np.pi / 180.0

# Convention A and B with propagation
reco_A_prop = []
reco_B_prop = []

for z_u, u_val, z_v, v_val in matched_pairs:
    z_avg = (z_u + z_v) / 2.0
    
    # Propagate u and v to z_avg
    u_prop = u_val + slope_u * (z_avg - z_u)
    v_prop = v_val + slope_v * (z_avg - z_v)
    
    # Convention A
    xA = (u_prop + v_prop) / (2.0 * np.cos(theta / 2.0))
    yA = (u_prop - v_prop) / (2.0 * np.sin(theta / 2.0))
    reco_A_prop.append((xA, yA, z_avg))
    
    # Convention B
    xB = u_prop
    yB = (v_prop - u_prop * np.cos(theta)) / np.sin(theta)
    reco_B_prop.append((xB, yB, z_avg))

# Compare with actual Reco Track 0 hits
reco_track0 = ak.to_numpy(rc_arr["TrackHitPos"][event_id][0])
mask_reco0 = (reco_track0[:, 0] > -9e8)
actual_reco0 = reco_track0[mask_reco0]

diffs_A = []
diffs_B = []
for pA in reco_A_prop:
    z_dist = np.abs(actual_reco0[:, 2] - pA[2])
    idx = np.argmin(z_dist)
    if z_dist[idx] < 50.0:
        true_x, true_y, true_z = actual_reco0[idx]
        diffs_A.append((pA[0] - true_x, pA[1] - true_y, pA[2] - true_z))

for pB in reco_B_prop:
    z_dist = np.abs(actual_reco0[:, 2] - pB[2])
    idx = np.argmin(z_dist)
    if z_dist[idx] < 50.0:
        true_x, true_y, true_z = actual_reco0[idx]
        diffs_B.append((pB[0] - true_x, pB[1] - true_y, pB[2] - true_z))

if diffs_A:
    diffs_A = np.array(diffs_A)
    print(f"\nConvention A with propagation residuals:")
    print(f"  dx: {np.mean(diffs_A[:, 0]):.2f} +/- {np.std(diffs_A[:, 0]):.2f}")
    print(f"  dy: {np.mean(diffs_A[:, 1]):.2f} +/- {np.std(diffs_A[:, 1]):.2f}")
    print(f"  dz: {np.mean(diffs_A[:, 2]):.2f} +/- {np.std(diffs_A[:, 2]):.2f}")

if diffs_B:
    diffs_B = np.array(diffs_B)
    print(f"\nConvention B with propagation residuals:")
    print(f"  dx: {np.mean(diffs_B[:, 0]):.2f} +/- {np.std(diffs_B[:, 0]):.2f}")
    print(f"  dy: {np.mean(diffs_B[:, 1]):.2f} +/- {np.std(diffs_B[:, 1]):.2f}")
    print(f"  dz: {np.mean(diffs_B[:, 2]):.2f} +/- {np.std(diffs_B[:, 2]):.2f}")
