import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from pathlib import Path

# Turn off interactive plotting for speed and memory efficiency
plt.ioff()

# 1. Open the ROOT file
file_path = Path("../Cut2.root")
if not file_path.exists():
    file_path = Path("Cut2.root")
if not file_path.exists():
    raise FileNotFoundError("Could not find Cut2.root")

print(f"Opening ROOT file: {file_path}")
f = uproot.open(file_path)
t_lc = f["Line_Candidates;2"]

n_events = t_lc.num_entries
print(f"Total events in ROOT file: {n_events}")

# Create output folder
output_dir = Path("event_displays")
output_dir.mkdir(exist_ok=True)
print(f"Created folder: {output_dir.resolve()}")

# Read all fields
lc_fields = [
    "nLinesU", "nLinesV", 
    "SlopeU", "InterceptU", "SlopeV", "InterceptV",
    "TrackHitPosU", "TrackHitPosV"
]
lc_arr = t_lc.arrays(lc_fields)

colors = ['blue', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

print("Starting batch event display generation...")
for event_id in range(n_events):
    n_u = lc_arr["nLinesU"][event_id]
    n_v = lc_arr["nLinesV"][event_id]
    
    # We only plot events that have at least one reconstructed track line or active hits
    # to avoid saving empty plots
    if n_u == 0 and n_v == 0:
        continue
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- 1. Left Panel: U View ---
    ax_u = axes[0]
    slopes_u = lc_arr["SlopeU"][event_id].tolist()
    intercepts_u = lc_arr["InterceptU"][event_id].tolist()
    track_hits_u = lc_arr["TrackHitPosU"][event_id]
    
    for i in range(n_u):
        hits = ak.to_numpy(track_hits_u[i])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        
        if len(act_hits) > 0:
            c = colors[i % len(colors)]
            ax_u.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=40, edgecolors='k', label=f"Track {i} Hits")
            
            z_min, z_max = np.min(act_hits[:, 0]), np.max(act_hits[:, 0])
            z_vals = np.linspace(z_min, z_max, 100)
            u_vals = slopes_u[i] * z_vals + intercepts_u[i]
            ax_u.plot(z_vals, u_vals, color=c, linestyle='-', lw=2, label=f"Track {i} Line")
            
    ax_u.set_title("U View (Z vs U)")
    ax_u.set_xlabel("Z Position (mm)")
    ax_u.set_ylabel("U Position (mm)")
    ax_u.grid(True, linestyle=':', alpha=0.6)
    if n_u > 0:
        ax_u.legend(loc='best')
        
    # --- 2. Right Panel: V View ---
    ax_v = axes[1]
    slopes_v = lc_arr["SlopeV"][event_id].tolist()
    intercepts_v = lc_arr["InterceptV"][event_id].tolist()
    track_hits_v = lc_arr["TrackHitPosV"][event_id]
    
    for j in range(n_v):
        hits = ak.to_numpy(track_hits_v[j])
        mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
        act_hits = hits[mask]
        
        if len(act_hits) > 0:
            c = colors[j % len(colors)]
            ax_v.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=40, edgecolors='k', label=f"Track {j} Hits")
            
            z_min, z_max = np.min(act_hits[:, 0]), np.max(act_hits[:, 0])
            z_vals = np.linspace(z_min, z_max, 100)
            v_vals = slopes_v[j] * z_vals + intercepts_v[j]
            ax_v.plot(z_vals, v_vals, color=c, linestyle='-', lw=2, label=f"Track {j} Line")
            
    ax_v.set_title("V View (Z vs V)")
    ax_v.set_xlabel("Z Position (mm)")
    ax_v.set_ylabel("V Position (mm)")
    ax_v.grid(True, linestyle=':', alpha=0.6)
    if n_v > 0:
        ax_v.legend(loc='best')
        
    plt.suptitle(f"TMS 2D Event Display: Event {event_id}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / f"event_{event_id}_uv.png")
    plt.close(fig)

print(f"Finished generating event displays. Saved to: {output_dir.resolve()}")
