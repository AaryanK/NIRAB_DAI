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

# Create output directory
output_dir = Path("amalgamated_samples")
output_dir.mkdir(exist_ok=True)
print(f"Created folder: {output_dir.resolve()}")

# Stereo transformation constants
theta = 3.0 * np.pi / 180.0
cos_half = np.cos(theta / 2.0)
sin_half = np.sin(theta / 2.0)

# Colors for plotting
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

# Sample 10 events (Events 0 to 9)
sample_event_ids = list(range(10))

for event_id in sample_event_ids:
    # Load Reco_Tree variables
    reco_arr = t_reco.arrays(["nTracks", "StartPos", "EndPos"], entry_start=event_id, entry_stop=event_id + 1)
    n_tracks = reco_arr["nTracks"][0]
    start_pos = ak.to_numpy(reco_arr["StartPos"][0])
    end_pos = ak.to_numpy(reco_arr["EndPos"][0])
    
    # Load Line_Candidates variables
    lc_arr = t_lc.arrays(["nLinesU", "nLinesV", "TrackHitPosU", "TrackHitPosV"], entry_start=event_id, entry_stop=event_id + 1)
    n_lines_u = lc_arr["nLinesU"][0]
    n_lines_v = lc_arr["nLinesV"][0]
    track_hits_u = lc_arr["TrackHitPosU"][0]
    track_hits_v = lc_arr["TrackHitPosV"][0]
    
    # Load Truth_Info variables (True Neutrino Vertex)
    tr_arr = t_tr.arrays(["NeutrinoX4"], entry_start=event_id, entry_stop=event_id + 1)
    nx4 = tr_arr["NeutrinoX4"][0].tolist()
    true_x = nx4[0] * 1000.0
    true_y = nx4[1] * 1000.0
    true_z = nx4[2] * 1000.0
    
    print(f"Generating plot for Event {event_id}...")
    
    # Create a 1x3 panel plot
    fig = plt.figure(figsize=(18, 6))
    
    # --- Panel 1: 3D Event Display (StartPos, EndPos & True Vertex) ---
    ax3d = fig.add_subplot(131, projection='3d')
    
    # Plot reconstructed tracks
    for i in range(n_tracks):
        xs, ys, zs = start_pos[i][0], start_pos[i][1], start_pos[i][2]
        xe, ye, ze = end_pos[i][0], end_pos[i][1], end_pos[i][2]
        c = colors[i % len(colors)]
        ax3d.plot([zs, ze], [xs, xe], [ys, ye], color=c, lw=3, label=f"Reco Track {i}")
        ax3d.scatter([zs], [xs], [ys], color='red', marker='o', s=40, zorder=5, label="Reco Start" if i == 0 else "")
        
    # Plot true neutrino interaction vertex
    ax3d.scatter([true_z], [true_x], [true_y], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex")
    
    ax3d.set_title("3D Reco Tracks & True Vertex")
    ax3d.set_xlabel("Z (mm)")
    ax3d.set_ylabel("X (mm)")
    ax3d.set_zlabel("Y (mm)")
    ax3d.grid(True, linestyle=':', alpha=0.6)
    ax3d.legend()
    
    # --- Panel 2: U-View (TrackHitsU, Projected Tracks & Projected True Vertex) ---
    ax_u = fig.add_subplot(132)
    
    # Plot U hits
    for i in range(n_lines_u):
        hits = ak.to_numpy(track_hits_u[i])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        if len(act_hits) > 0:
            c = colors[i % len(colors)]
            ax_u.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=30, alpha=0.7, edgecolors='k', label=f"U-Hit Cluster {i}")
            
    # Plot projected tracks
    for i in range(n_tracks):
        xs, ys, zs = start_pos[i][0], start_pos[i][1], start_pos[i][2]
        xe, ye, ze = end_pos[i][0], end_pos[i][1], end_pos[i][2]
        us = xs * cos_half + ys * sin_half
        ue = xe * cos_half + ye * sin_half
        ax_u.plot([zs, ze], [us, ue], color='black', linestyle='--', lw=2, label="Reco Track Proj" if i == 0 else "")
        
    # Plot projected true vertex: u_true = x_true * cos + y_true * sin
    true_u = true_x * cos_half + true_y * sin_half
    ax_u.scatter([true_z], [true_u], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex Proj")
    
    ax_u.set_title("U View: Hits & Projections")
    ax_u.set_xlabel("Z Position (mm)")
    ax_u.set_ylabel("U Position (mm)")
    ax_u.grid(True, linestyle=':', alpha=0.6)
    ax_u.legend(loc='best')
    
    # --- Panel 3: V-View (TrackHitsV, Projected Tracks & Projected True Vertex) ---
    ax_v = fig.add_subplot(133)
    
    # Plot V hits
    for j in range(n_lines_v):
        hits = ak.to_numpy(track_hits_v[j])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        if len(act_hits) > 0:
            c = colors[j % len(colors)]
            ax_v.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=30, alpha=0.7, edgecolors='k', label=f"V-Hit Cluster {j}")
            
    # Plot projected tracks
    for j in range(n_tracks):
        xs, ys, zs = start_pos[j][0], start_pos[j][1], start_pos[j][2]
        xe, ye, ze = end_pos[j][0], end_pos[j][1], end_pos[j][2]
        vs = xs * cos_half - ys * sin_half
        ve = xe * cos_half - ye * sin_half
        ax_v.plot([zs, ze], [vs, ve], color='black', linestyle='--', lw=2, label="Reco Track Proj" if j == 0 else "")
        
    # Plot projected true vertex: v_true = x_true * cos - y_true * sin
    true_v = true_x * cos_half - true_y * sin_half
    ax_v.scatter([true_z], [true_v], color='gold', marker='*', s=150, edgecolors='black', zorder=10, label="True Vertex Proj")
    
    ax_v.set_title("V View: Hits & Projections")
    ax_v.set_xlabel("Z Position (mm)")
    ax_v.set_ylabel("V Position (mm)")
    ax_v.grid(True, linestyle=':', alpha=0.6)
    ax_v.legend(loc='best')
    
    plt.suptitle(f"Amalgamated Display (with True Vertex): Event {event_id}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / f"amalgamated_event_{event_id}.png")
    plt.close()

print(f"Successfully generated 10 sample event displays in: {output_dir.resolve()}")
