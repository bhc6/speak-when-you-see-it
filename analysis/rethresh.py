"""Exact offline re-thresholding of the MISTAKE decision of a qic_prob run (order done_first).
In done_first the completion decision never depends on the mistake decision, so the sequence of checks is identical
for every tau_mis >= the logged tau; raising tau_mis only removes Feedback entries. Each Feedback in the shard
predictions is matched, in order, to the next logged mistake call with yes=True (asserted 1:1).
python rethresh.py OUT.json TAU shard0.json plog0.jsonl [shard1.json plog1.jsonl ...]  -> prints #feedback kept"""
import json, sys
out, tau = sys.argv[1], float(sys.argv[2]); args = sys.argv[3:]; P = []; kept = tot = 0
for sj, pl in zip(args[::2], args[1::2]):
    S = json.load(open(sj)); L = [json.loads(l) for l in open(pl)]
    yes = [r for r in L if r["mode"] == "mistake_inference" and r["yes"]]
    assert all(r["tau"] <= tau + 1e-9 for r in L if r["mode"] == "mistake_inference"), "can only raise tau_mis"
    nfb = sum(t.startswith("Feedback") for p in S for t in p["pred_texts"]); assert nfb == len(yes), (sj, nfb, len(yes))
    it = iter(yes)
    for p in S:
        T, TS = [], []
        for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
            if t.startswith("Feedback"):
                r = next(it); tot += 1
                if r["p"] < tau: continue
                kept += 1
            T.append(t); TS.append(ts)
        P.append({"video_id": p["video_id"], "pred_texts": T, "pred_timestamps": TS})
v = [p["video_id"] for p in P]; assert len(v) == len(set(v))
json.dump(P, open(out, "w")); print(f"tau_mis {tau}: feedback kept {kept}/{tot}")
