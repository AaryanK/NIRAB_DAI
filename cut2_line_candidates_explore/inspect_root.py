import uproot
import awkward as ak
import numpy as np

file_path = "Cut2.root"

try:
    f = uproot.open(file_path)
    print("ROOT file opened successfully.")
except Exception as e:
    print(f"Error opening ROOT file: {e}")
    exit(1)

print("\n--- Keys in file ---")
for k in f.keys():
    print(k, type(f[k]))

print("\n--- Checking for Line_Candidates ---")
lc_keys = [k for k in f.keys() if "Line_Candidates" in k]
if not lc_keys:
    print("No Line_Candidates tree found!")
    exit(1)

tree_name = lc_keys[0]
print(f"Using tree key: {tree_name}")
tree = f[tree_name]
print(f"Number of entries: {tree.num_entries}")

print("\n--- Checking branches ---")
available = set(tree.keys())

wanted = [
    "TrackHitU", "TrackHitV",
    "nHitsInTrackU", "nHitsInTrackV",
    "FirstHoughHitU", "FirstHoughHitV",
    "LastHoughHitU", "LastHoughHitV",
    "TrackLengthU", "TrackLengthV",
    "SlopeU", "SlopeV",
    "InterceptU", "InterceptV",
    "RecoHitPos", "RecoHitEnergy", "RecoHitPE",
    "RecoHitBar", "RecoHitPlane", "RecoHitSlice"
]

found = []
missing = []
for b in wanted:
    if b in available:
        found.append(b)
    else:
        missing.append(b)

print(f"Found branches ({len(found)}):", found)
print(f"Missing branches ({len(missing)}):", missing)

print("\n--- Detailed branch structures (first entry) ---")
# Load some data
to_load = [b for b in found]
arr = tree.arrays(to_load, entry_stop=5)
for name in arr.fields:
    print(f"\nBranch: {name}")
    print(f"  Type: {ak.type(arr[name])}")
    print(f"  Data (first 3 entries): {arr[name][:3].tolist()}")

print("\n--- Summary Stats ---")
# Let's read 5000 entries to compute some quick typical stats
arr_stats = tree.arrays([b for b in found if b in ["nHitsInTrackU", "nHitsInTrackV", "TrackLengthU", "TrackLengthV"]], entry_stop=5000)
for field in arr_stats.fields:
    try:
        flat = ak.flatten(arr_stats[field], axis=None)
        if len(flat) > 0:
            print(f"  {field}: mean={np.mean(flat):.2f}, min={np.min(flat):.2f}, max={np.max(flat):.2f}")
    except Exception as e:
        print(f"  Error summarizing {field}: {e}")
