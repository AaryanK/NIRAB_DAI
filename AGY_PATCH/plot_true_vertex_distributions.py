import uproot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set plotting style
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (7, 6),
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
t_tr = f["Truth_Info;2"]

n_events = min(1000, t_tr.num_entries)
tr_arr = t_tr.arrays(["NeutrinoX4"], entry_stop=n_events)

# Extract coordinates and scale to mm
nx4 = tr_arr["NeutrinoX4"].to_numpy()
true_x = nx4[:, 0] * 1000.0
true_y = nx4[:, 1] * 1000.0
true_z = nx4[:, 2] * 1000.0

# Filter for TMS fiducial volume
tms_mask = (true_z >= 11124.0) & (true_z <= 18544.0)
tms_x = true_x[tms_mask]
tms_y = true_y[tms_mask]
tms_z = true_z[tms_mask]

print(f"Total events: {n_events}")
print(f"Interactions inside TMS: {len(tms_z)}")

# Helper function to style plots
def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=10)

# --- 1. True X Distribution ---
fig, ax = plt.subplots()
ax.hist(tms_x, bins=30, color='royalblue', edgecolor='black', alpha=0.8)
style_plot(ax, "True X coordinate distribution (TMS)", "True X (mm)", "Counts")
plt.tight_layout()
plt.savefig("true_x_distribution.png")
plt.close()

# --- 2. True Y Distribution ---
fig, ax = plt.subplots()
ax.hist(tms_y, bins=30, color='seagreen', edgecolor='black', alpha=0.8)
style_plot(ax, "True Y coordinate distribution (TMS)", "True Y (mm)", "Counts")
plt.tight_layout()
plt.savefig("true_y_distribution.png")
plt.close()

# --- 3. True Z Distribution ---
fig, ax = plt.subplots()
ax.hist(tms_z, bins=30, color='crimson', edgecolor='black', alpha=0.8)
style_plot(ax, "True Z coordinate distribution (TMS)", "True Z (mm)", "Counts")
plt.tight_layout()
plt.savefig("true_z_distribution.png")
plt.close()

print("Successfully saved true coordinate distribution plots:")
print(" - true_x_distribution.png\n - true_y_distribution.png\n - true_z_distribution.png")
