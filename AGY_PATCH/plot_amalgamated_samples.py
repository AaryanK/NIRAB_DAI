import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import awkward as ak
from pathlib import Path

# Set plotting style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
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
t_lc = f["Line_Candidates;2"]
t_tr = f["Truth_Info;2"]

output_dir = Path("amalgamated_samples")
output_dir.mkdir(exist_ok=True)

# New stereo tilt angle: phi = 3 degrees
phi = 3.0 * np.pi / 180.0
cos_phi = np.cos(phi)
sin_phi = np.sin(phi)

colors = ['blue', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']
sample_event_ids = list(range(10))

for event_id in sample_event_ids:
    reco_arr = t_reco.arrays(["nTracks", "StartPos", "EndPos"], entry_start=event_id, entry_stop=event_id + 1)
    n_tracks = reco_arr["nTracks"][0]
    start_pos = ak.to_numpy(reco_arr["StartPos"][0])
    end_pos = ak.to_numpy(reco_arr["EndPos"][0])
    
    lc_arr = t_lc.arrays(["nLinesU", "nLinesV", "TrackHitPosU", "TrackHitPosV"], entry_start=event_id, entry_stop=event_id + 1)
    n_lines_u = lc_arr["nLinesU"][0]
    n_lines_v = lc_arr["nLinesV"][0]
    track_hits_u = lc_arr["TrackHitPosU"][0]
    track_hits_v = lc_arr["TrackHitPosV"][0]
    
    tr_arr = t_tr.arrays(["NeutrinoX4"], entry_start=event_id, entry_stop=event_id + 1)
    nx4 = tr_arr["NeutrinoX4"][0].tolist()
    true_x = nx4[0] * 1000.0
    true_y = nx4[1] * 1000.0
    true_z = nx4[2] * 1000.0
    
    fig = plt.figure(figsize=(18, 6))
    
    # --- Panel 1: 3D Display ---
    ax3d = fig.add_subplot(131, projection='3d')
    for i in range(n_tracks):
        xs, ys, zs = start_pos[i][0], start_pos[i][1], start_pos[i][2]
        xe, ye, ze = end_pos[i][0], end_pos[i][1], end_pos[i][2]
        c = colors[i % len(colors)]
        ax3d.plot([zs, ze], [xs, xe], [ys, ye], color=c, lw=3, label=f"Reco Track {i}")
        ax3d.scatter([zs], [xs], [ys], color='red', marker='o', s=40, zorder=5, label="Reco Start" if i == 0 else "")
    ax3d.scatter([true_z], [true_x], [true_y], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex")
    ax3d.set_title("3D Reco Tracks & True Vertex")
    ax3d.set_xlabel("Z (mm)")
    ax3d.set_ylabel("X (mm)")
    ax3d.set_zlabel("Y (mm)")
    ax3d.grid(True, linestyle=':', alpha=0.6)
    ax3d.legend()
    
    # --- Panel 2: U-View ---
    ax_u = fig.add_subplot(132)
    for i in range(n_lines_u):
        hits = ak.to_numpy(track_hits_u[i])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        if len(act_hits) > 0:
            c = colors[i % len(colors)]
            ax_u.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=30, alpha=0.7, edgecolors='k', label=f"U-Hit Cluster {i}")
    for i in range(n_tracks):
        xs, ys, zs = start_pos[i][0], start_pos[i][1], start_pos[i][2]
        xe, ye, ze = end_pos[i][0], end_pos[i][1], end_pos[i][2]
        # Project to U using new phi = 3 degrees
        us = xs * cos_phi + ys * sin_phi
        ue = xe * cos_phi + ye * sin_phi
        ax_u.plot([zs, ze], [us, ue], color='black', linestyle='--', lw=2, label="Reco Track Proj" if i == 0 else "")
    true_u = true_x * cos_phi + true_y * sin_phi
    ax_u.scatter([true_z], [true_u], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex Proj")
    ax_u.set_title("U View: Hits & Projections (phi = 3 deg)")
    ax_u.set_xlabel("Z Position (mm)")
    ax_u.set_ylabel("U Position (mm)")
    ax_u.grid(True, linestyle=':', alpha=0.6)
    ax_u.legend(loc='best')
    
    # --- Panel 3: V-View ---
    ax_v = fig.add_subplot(133)
    for j in range(n_lines_v):
        hits = ak.to_numpy(track_hits_v[j])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        if len(act_hits) > 0:
            c = colors[j % len(colors)]
            ax_v.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=30, alpha=0.7, edgecolors='k', label=f"V-Hit Cluster {j}")
    for j in range(n_tracks):
        xs, ys, zs = start_pos[j][0], start_pos[j][1], start_pos[j][2]
        xe, ye, ze = end_pos[j][0], end_pos[j][1], end_pos[j][2]
        # Project to V using new phi = 3 degrees
        vs = xs * cos_phi - ys * sin_phi
        ve = xe * cos_phi - ye * sin_phi
        ax_v.plot([zs, ze], [vs, ve], color='black', linestyle='--', lw=2, label="Reco Track Proj" if j == 0 else "")
    true_v = true_x * cos_phi - true_y * sin_phi
    ax_v.scatter([true_z], [true_v], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex Proj")
    ax_v.set_title("V View: Hits & Projections (phi = 3 deg)")
    ax_v.set_xlabel("Z Position (mm)")
    ax_v.set_ylabel("V Position (mm)")
    ax_v.grid(True, linestyle=':', alpha=0.6)
    ax_v.legend(loc='best')
    
    plt.suptitle(f"Amalgamated Display (phi = 3 deg): Event {event_id}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / f"amalgamated_event_{event_id}.png")
    plt.close()

print("Successfully generated 10 sample event displays with phi = 3 degrees.")
