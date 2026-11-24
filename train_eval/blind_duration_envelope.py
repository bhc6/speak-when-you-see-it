"""(1) Duration-prior blind rule: complete when elapsed >= c * (median TRAIN duration of this instruction; fallback: global
train median), optional one feedback at elapsed >= B for instructions with train mistake rate >= r. Uses only instruction
text, step start and elapsed time. c, B, r selected on validation (max F1, ties IC), applied once to test; also reports the
best test point per IC level (favourable to blind side).
(2) Upper envelope of the blind frontier over ALL single rules (time/prior grid + duration rules): concave hull of (IC, F1)
vertices, then each hull edge evaluated as a real per-step mixture (official eval).
python blind_duration_envelope.py EVAL_DIR DV SPLIT"""
import sys, json, hashlib, statistics as st, collections
EV, DV, SP = sys.argv[1:4]; sys.argv = [sys.argv[0], EV]
exec(open("WORKSPACE/qic/experiments/plan_repr/blind_baselines.py").read().split("val = utils.load_dataset")[0])
dur = collections.defaultdict(list)
tr = utils.load_dataset("neurips2025", PS, "train")
for a in tr.annotations:
    g = ev.get_ground_truth_data(tr, a["video_id"])
    for tx, ts in zip(g["texts"], g["texts_timestamps"]): dur[norm(tx[0])].append(max(ts) - ts[0])
gmed = st.median([d for v in dur.values() for d in v])
med = lambda s: st.median(dur[norm(s)]) if dur.get(norm(s)) else gmed
def gen(items, pol):
    out = []
    for item in items:
        texts, ts = [], []
        for k, (instr, start, last) in enumerate(item["steps"]):
            kind, A, B, r = pol(item["video_id"], k)
            A = A * med(instr) if kind == "dur" else A
            texts.append("Instruction: " + instr); ts.append(start)
            init = int(start * 2); j = 1; fb = False; want = B is not None and (r is None or ((mrate(instr) or 0) >= r))
            while True:
                t = (init + 10 * j) / 2; e = t - start
                if t > last: break
                if e >= A: texts.append("Success: Good job!"); ts.append(t); break
                if want and not fb and e >= B: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
        out.append({"video_id": item["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return out
def grid():
    for c in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0]:
        for B in [None, 15, 25]:
            for r in ([None] if B is None else [None, 0.3]): yield ("dur", c, B, r)
    for A in range(10, 181, 10):
        for B in [None] + list(range(5, 181, 10)):
            if B is not None and B >= A: continue
            for r in ([None] if B is None else [None, 0.2, 0.4, 0.6]): yield ("time", A, B, r)
def run(ds, items, pol): return score(ds, gen(items, pol))
val = utils.load_dataset("neurips2025", PS, "validation") if DV == "neurips2025" else None
ds = utils.load_dataset(DV, PS if DV == "neurips2025" else "advanced_planning", SP); items = ds_items(ds)
res = {"gmed": gmed, "n_instr_with_prior": None}
cov = sum(bool(dur.get(norm(i))) for it in items for i, _, _ in it["steps"]); tot = sum(len(it["steps"]) for it in items)
res["n_instr_with_prior"] = f"{cov}/{tot}"
if val is not None:
    vitems = ds_items(val); vg = [(p, run(val, vitems, lambda v, k, p=p: p)) for p in grid() if p[0] == "dur"]
    sel = max(vg, key=lambda g: (g[1]["F1"], g[1]["ic"])); selic = max(vg, key=lambda g: (g[1]["ic"], g[1]["F1"]))
    res["dur_val_selected"] = [sel[0], sel[1], run(ds, items, lambda v, k: sel[0])]
    res["dur_val_maxic"] = [selic[0], selic[1], run(ds, items, lambda v, k: selic[0])]
pts = [(p, run(ds, items, lambda v, k, p=p: p)) for p in grid()]
res["dur_test_points"] = [(p, s["ic"], s["F1"]) for p, s in pts if p[0] == "dur"]
# concave upper hull over (IC, F1)
P = sorted({(s["ic"], s["F1"], p) for p, s in pts}, key=lambda x: (x[0], -x[1]))
hull = []
for x in sorted(P, key=lambda x: (x[0], x[1])):
    while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (x[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (x[0] - hull[-2][0]) >= 0: hull.pop()
    hull.append(x)
top = max(hull, key=lambda x: x[1]); hull = [h for h in hull if h[0] >= top[0]]
res["hull_vertices"] = [(h[0], h[1], h[2]) for h in hull]
def hh(v, k): return int(hashlib.md5(f"{v}/{k}".encode()).hexdigest(), 16) / 16 ** 32
mix = []
for a, b in zip(hull, hull[1:]):
    for q in [0.25, 0.5, 0.75]:
        s = run(ds, items, lambda v, k, a=a, b=b, q=q: b[2] if hh(v, k) < q else a[2]); mix.append((a[2], b[2], q, s["ic"], s["F1"]))
res["hull_mixtures"] = mix
json.dump(res, open(f"WORKSPACE/qic/results/blind_envelope_{DV}{TAG}_{SP}.json", "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in res if k not in ("dur_test_points", "hull_mixtures")}, default=str))
