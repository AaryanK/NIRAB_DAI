import uproot
import awkward as ak

f = uproot.open("../Cut2.root")
t_tr = f["Truth_Info;2"]
vtx_ids = t_tr["RecoTrackPrimaryParticleVtxId"].array()

shared_vtx_events = 0
events_with_2plus_tracks = 0

for idx, ev_ids in enumerate(vtx_ids):
    list_ids = ev_ids.tolist() if hasattr(ev_ids, "tolist") else list(ev_ids)
    if len(list_ids) >= 2:
        events_with_2plus_tracks += 1
        # Check if there is any duplicated ID
        if len(set(list_ids)) < len(list_ids):
            shared_vtx_events += 1
            if shared_vtx_events <= 5:
                print(f"Event {idx}: RecoTrackPrimaryParticleVtxId = {list_ids}")

print(f"\nEvents with >=2 tracks: {events_with_2plus_tracks}")
print(f"Events with shared vertex tracks: {shared_vtx_events}")
