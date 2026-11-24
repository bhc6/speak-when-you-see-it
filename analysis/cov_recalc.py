"""Coverage audit 11-17: per-video IC-Acc and the share of unreachable IC denominators. Same step logic as decision_decomp.py
(official eval semantics). cwd = official eval repo. python cov_recalc.py OUT.json DS:NAME=PRED.json ..."""
import json, sys, collections, statistics
from utils import load_dataset
from eval import check_if_correct_instruction_given
GTS = {}
def gt(dv):
    if dv not in GTS:
        ds = load_dataset(dv, "advanced_planning", "test"); GTS[dv] = {ds[i]["video_id"]: ds[i] for i in range(len(ds))}
    return GTS[dv]
def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruct"): out.append([])
        if out: out[-1].append((t, ts))
    return out
res = {"denominator": {}, "runs": {}}
for dv in ("neurips2025", "ego_mc_bench"):
    G = gt(dv); n = single = last_single = 0
    for v, g in G.items():
        for i in range(len(g["texts"])):
            if g["has_mistake"][i]: continue
            n += 1
            if len(g["texts_timestamps"][i]) == 1:
                single += 1; last_single += (i == len(g["texts"]) - 1)
    res["denominator"][dv] = dict(videos=len(G), clean_steps=n, clean_steps_with_instruction_only=single, of_which_last_step_of_video=last_single)
for arg in sys.argv[2:]:
    key, path = arg.split("=", 1); dv, name = key.split(":"); G = gt(dv); pv = {}; hit_single = 0
    for p in json.load(open(path)):
        g = G[p["video_id"]]; S = segs(p); h = n = 0
        for i in range(len(g["texts"])):
            if g["has_mistake"][i]: continue
            n += 1; ts_i = g["texts_timestamps"][i]; end = max(ts_i); seg = S[i] if i < len(S) else None
            ok = (seg is not None and len(seg) == 2 and seg[-1][0].startswith("Success") and abs(seg[-1][1] - end) < 15
                  and check_if_correct_instruction_given(g["texts"][i][0], seg[0][0].replace("Instruct:", "").strip()))
            h += ok; hit_single += (ok and len(ts_i) == 1)
        if n: pv[p["video_id"]] = (h, n)
    H = sum(h for h, n in pv.values()); N = sum(n for h, n in pv.values()); r = [h / n for h, n in pv.values()]
    res["runs"][f"{dv}:{name}"] = dict(videos_with_clean_steps=len(pv), ic=round(100 * H / N, 1), hits=H, clean_steps=N, per_video_median=round(statistics.median(r), 3),
                                       videos_with_zero=sum(x == 0 for x in r), hits_on_instruction_only_steps=hit_single)
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
