import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from pathlib import Path

# Set plotting style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

# 1. Open the ROOT file
file_path = Path("../Cut2.root")
if not file_path.exists():
    file_path = Path("Cut2.root")
if not file_path.exists():
    raise FileNotFoundError("Could not find Cut2.root")

print(f"Opening ROOT file: {file_path}")
f = uproot.open(file_path)
t_lc = f["Line_Candidates;2"]

# Read event 0
event_id = 0
lc_fields = [
    "nLinesU", "nLinesV", 
    "SlopeU", "InterceptU", "SlopeV", "InterceptV",
    "TrackHitPosU", "TrackHitPosV"
]
lc_arr = t_lc.arrays(lc_fields, entry_start=event_id, entry_stop=event_id + 1)

n_u = lc_arr["nLinesU"][0]
n_v = lc_arr["nLinesV"][0]
print(f"Event {event_id}: {n_u} U-view lines, {n_v} V-view lines.")

# Simple color list for tracks
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan']

# --- 1. Plot U View ---
fig, ax = plt.subplots(figsize=(10, 6))

slopes_u = lc_arr["SlopeU"][0].tolist()
intercepts_u = lc_arr["InterceptU"][0].tolist()
track_hits_u = lc_arr["TrackHitPosU"][0]

# Plot U-view hits and fitted Hough lines
for i in range(n_u):
    hits = ak.to_numpy(track_hits_u[i])
    mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
    act_hits = hits[mask]
    
    if len(act_hits) > 0:
        c = colors[i % len(colors)]
        # Plot hits
        ax.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=40, edgecolors='k', label=f"Track {i} Hits")
        
        # Plot fitted line
        z_min, z_max = np.min(act_hits[:, 0]), np.max(act_hits[:, 0])
        z_vals = np.linspace(z_min, z_max, 100)
        u_vals = slopes_u[i] * z_vals + intercepts_u[i]
        ax.plot(z_vals, u_vals, color=c, linestyle='-', lw=2, label=f"Track {i} Line")

ax.set_title(f"2D Event Display: U View (Event {event_id})")
ax.set_xlabel("Z Position (mm)")
ax.set_ylabel("U Position (mm)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig("u_view_event0.png")
plt.close()

# --- 2. Plot V View ---
fig, ax = plt.subplots(figsize=(10, 6))

slopes_v = lc_arr["SlopeV"][0].tolist()
intercepts_v = lc_arr["InterceptV"][0].tolist()
track_hits_v = lc_arr["TrackHitPosV"][0]

for j in range(n_v):
    hits = ak.to_numpy(track_hits_v[j])
    mask = (hits[:, 0] != 0.0) | (hits[:, 1] != 0.0)
    act_hits = hits[mask]
    
    if len(act_hits) > 0:
        c = colors[j % len(colors)]
        ax.scatter(act_hits[:, 0], act_hits[:, 1], color=c, marker='o', s=40, edgecolors='k', label=f"Track {j} Hits")
        
        z_min, z_max = np.min(act_hits[:, 0]), np.max(act_hits[:, 0])
        z_vals = np.linspace(z_min, z_max, 100)
        v_vals = slopes_v[j] * z_vals + intercepts_v[j]
        ax.plot(z_vals, v_vals, color=c, linestyle='-', lw=2, label=f"Track {j} Line")

ax.set_title(f"2D Event Display: V View (Event {event_id})")
ax.set_xlabel("Z Position (mm)")
ax.set_ylabel("V Position (mm)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig("v_view_event0.png")
plt.close()

print("Successfully saved 2D stereo view event displays:")
print(" - u_view_event0.png\n - v_view_event0.png")
