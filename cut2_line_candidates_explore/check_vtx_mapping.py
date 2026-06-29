import uproot
import awkward as ak

f = uproot.open("../Cut2.root")
t_tr = f["Truth_Info;2"]

vtx_ids = t_tr["TrueVtxID"].array(entry_stop=5)
track_vtx_ids = t_tr["RecoTrackPrimaryParticleVtxId"].array(entry_stop=5)

for i in range(5):
    print(f"\nEvent {i}:")
    print(f"  TrueVtxID list (first 10): {vtx_ids[i][:10].tolist()}")
    print(f"  TrueVtxID list size: {len(vtx_ids[i])}")
    print(f"  Track Vtx IDs: {track_vtx_ids[i].tolist()}")
    # Check if Track Vtx IDs are in TrueVtxID list
    for tv_id in track_vtx_ids[i].tolist():
        if tv_id in vtx_ids[i].tolist():
            idx = vtx_ids[i].tolist().index(tv_id)
            print(f"    Vtx ID {tv_id} found at index {idx}")
        else:
            print(f"    Vtx ID {tv_id} NOT FOUND in TrueVtxID list!")
