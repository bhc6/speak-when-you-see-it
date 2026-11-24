"""Bootstrap confidence for 'system lies beyond the video-blind frontier' (CPU). For every blind policy (time/prior/
duration rule grid + learned blind policy from learned.json) and every system prediction file, per-video official-eval
counts; then 2000 video resamples: frontier = concave upper hull of all blind policies' (IC, F1) on the resample;
margin = F1_sys - frontier(IC_sys). Reports P(margin > 0) and the 95% CI of the margin.
python frontier_boot.py EVAL_DIR DV LEARNED_JSON name=pred.json ..."""
import sys, os, io, re, json, random, contextlib, statistics as st, collections, types
_bs = types.ModuleType("bert_score"); _bs.BERTScorer = lambda *a, **k: None; sys.modules["bert_score"] = _bs
EV, DV = sys.argv[1:3]; LJ = os.path.abspath(sys.argv[3]); SYS = [(n, os.path.abspath(f)) for n, f in (a.split("=", 1) for a in sys.argv[4:])]; sys.argv = [sys.argv[0], EV]
exec(open("WORKSPACE/qic/experiments/plan_repr/blind_baselines.py").read().split("val = utils.load_dataset")[0])
dur = collections.defaultdict(list); tr = utils.load_dataset("neurips2025", "advanced_planning", "train")
for a in tr.annotations:
    g = ev.get_ground_truth_data(tr, a["video_id"])
    for tx, ts in zip(g["texts"], g["texts_timestamps"]): dur[norm(tx[0])].append(max(ts) - ts[0])
gmed = st.median([d for v in dur.values() for d in v]); med = lambda s: st.median(dur[norm(s)]) if dur.get(norm(s)) else gmed
ds = utils.load_dataset(DV, "advanced_planning", "test"); items = ds_items(ds)
def gen(kind, A, B, r):
    out = []
    for item in items:
        texts, ts = [], []
        for instr, start, last in item["steps"]:
            AA = A * med(instr) if kind == "dur" else A
            texts.append("Instruction: " + instr); ts.append(start); init = int(start * 2); j = 1; fb = False
            want = B is not None and (r is None or ((mrate(instr) or 0) >= r))
            while True:
                t = (init + 10 * j) / 2; e = t - start
                if t > last: break
                if e >= AA: texts.append("Success: Good job!"); ts.append(t); break
                if want and not fb and e >= B: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
        out.append({"video_id": item["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return out
def per_video(P):
    cl = {p["video_id"]: sum(not x for x in ev.get_ground_truth_data(ds, p["video_id"])["has_mistake"]) for p in P}
    P = sorted(P, key=lambda p: (cl[p["video_id"]] == 0, p["video_id"])); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(ds, P)
    o = buf.getvalue(); C = [tuple(map(int, m)) for m in re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)]
    I = [float(x) for x in re.findall(r"IC-Acc: ([\d.]+)", o)]; res, prev, pdet, cum = {}, (0, 0, 0, 0), 0, 0
    for p, c, ic in zip(P, C, I):
        v = p["video_id"]; cum += cl[v]; det = round(ic * cum / 100)
        res[v] = [*(x - y for x, y in zip(c, prev)), det - pdet, cl[v]]; prev, pdet = c, det
    return res
pols = []
for A in range(10, 181, 10):
    for B in [None] + list(range(5, 181, 10)):
        if B is not None and B >= A: continue
        for r in ([None] if B is None else [None, 0.2, 0.4]): pols.append(("time", A, B, r))
for c in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0):
    for B in (None, 15, 25):
        for r in ([None] if B is None else [None, 0.3]): pols.append(("dur", c, B, r))
PV = [per_video(gen(*p)) for p in pols]
L = json.load(open(LJ)).get(f"{DV}_test", []); PV += [r["per_video"] for r in L]; nb_rule = len(pols)
S = {n: per_video(json.load(open(f))) for n, f in SYS}
vids = sorted(PV[0])
def agg(pv, V):
    tp, fp, tn, fn, det, cl = (sum(pv[v][i] for v in V) for i in range(6))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return 100 * det / max(cl, 1), 2 * p * r / max(p + r, 1e-9)
def hull_at(pts, ic):
    pts = sorted(set(pts)); h = []
    for x in pts:
        while len(h) >= 2 and (h[-1][0]-h[-2][0])*(x[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(x[0]-h[-2][0]) >= 0: h.pop()
        h.append(x)
    best = 0.0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in h if i >= ic])
full = [agg(pv, vids) for pv in PV]
print(DV, "blind policies", len(PV), f"(rules {nb_rule}, learned {len(PV) - nb_rule})", flush=True)
for n in S:
    ic, f1 = agg(S[n], vids); m0 = f1 - hull_at(full, ic)
    mr = f1 - hull_at(full[:nb_rule], ic); ml = f1 - hull_at(full[nb_rule:], ic) if len(PV) > nb_rule else float("nan")
    rng = random.Random(0); M = []
    for _ in range(2000):
        V = [rng.choice(vids) for _ in vids]; pts = [agg(pv, V) for pv in PV]; a, b = agg(S[n], V); M.append(b - hull_at(pts, a))
    M.sort(); print(f"{n}: IC {ic:.1f} F1 {f1:.3f} | margin vs all-blind {m0:+.3f} (rules only {mr:+.3f}, learned only {ml:+.3f}) | boot 95% [{M[50]:+.3f},{M[1949]:+.3f}] P(beyond)={sum(x > 0 for x in M) / 2000:.3f}", flush=True)
