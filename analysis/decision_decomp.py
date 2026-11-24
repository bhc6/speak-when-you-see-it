"""TIDE-style decomposition of proactive decisions (official per-step protocol, official eval semantics).
Clean (mistake-free) steps -> IC outcome: HIT | voided by false alarm | confirmed too early | too late | never confirmed | not reached.
Mistake side: TP / FN; FP split into 'on a clean step' vs 'on a mistake step but outside +-15 s'.
Usage (cwd = official eval repo): python decision_decomp.py DATASET_VERSION NAME=PRED_JSON ..."""
import json, sys, collections
from utils import load_dataset
from eval import check_if_correct_instruction_given
dv = sys.argv[1]; ds = load_dataset(dv, "advanced_planning", "test")
GT = {ds[i]["video_id"]: ds[i] for i in range(len(ds))}
def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruct"): out.append([])
        if out: out[-1].append((t, ts))
    return out
res = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1); preds = json.load(open(path)); c = collections.Counter(); early = []
    for p in preds:
        g = GT[p["video_id"]]; S = segs(p)
        for i in range(len(g["texts"])):
            ts_i = g["texts_timestamps"][i]; start, end = ts_i[0], max(ts_i)
            seg = S[i] if i < len(S) else None
            fbs = [(t, ts) for t, ts in seg if t.startswith("Feedback")] if seg else []
            if g["has_mistake"][i]:
                c["mis_steps"] += 1; c["mis_steps_flagged"] += bool(fbs); continue
            c["clean_steps"] += 1; c["clean_fb"] += len(fbs)
            if seg is None: c["not reached"] += 1; continue
            if len(seg) == 1: c["never confirmed"] += 1; continue
            if not check_if_correct_instruction_given(g["texts"][i][0], seg[0][0].replace("Instruct:", "").strip()): c["instruction mismatch"] += 1; continue
            last = seg[-1]
            if fbs: c["voided by false alarm"] += 1
            elif not last[0].startswith("Success"): c["never confirmed"] += 1
            elif abs(last[1] - end) < 15: c["HIT"] += 1
            elif last[1] < end: c["confirmed too early"] += 1; early.append(end - last[1])
            else: c["confirmed too late"] += 1
    n = c["clean_steps"]
    res[name] = {k: round(100 * c[k] / n, 1) for k in ("HIT", "voided by false alarm", "confirmed too early", "confirmed too late", "never confirmed", "not reached", "instruction mismatch")}
    res[name]["false alarms per clean step"] = round(c["clean_fb"] / n, 2)
    res[name]["mistake steps flagged %"] = round(100 * c["mis_steps_flagged"] / max(1, c["mis_steps"]), 1)
    res[name]["median early (s)"] = sorted(early)[len(early) // 2] if early else None
print(json.dumps(res, indent=1))
