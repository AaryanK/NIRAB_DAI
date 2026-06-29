import uproot
import awkward as ak

file_path = "../Cut2.root"
f = uproot.open(file_path)
tree = f["Line_Candidates;2"]
available = set(tree.keys())

wanted = [
    "TrackHitPosU", "TrackHitPosV",
    "nHitsInTrackU", "nHitsInTrackV",
    "TrackLengthU", "TrackLengthV",
    "SlopeU", "SlopeV",
    "InterceptU", "InterceptV",
    "FirstHoughHitU", "FirstHoughHitV",
    "LastHoughHitU", "LastHoughHitV",
    "RecoHitPos", "RecoHitEnergy", "RecoHitPE",
    "RecoHitBar", "RecoHitPlane", "RecoHitSlice"
]

print("Checking presence of branches:")
for b in wanted:
    print(f"  {b:20s}: {'FOUND' if b in available else 'MISSING'}")

# If they exist, let's print their types and some values
for b in ["TrackHitPosU", "TrackHitPosV"]:
    if b in available:
        data = tree[b].array(entry_stop=3)
        print(f"\nBranch: {b}")
        print(f"  Type: {ak.type(data)}")
        print(f"  Data: {data.tolist()}")
