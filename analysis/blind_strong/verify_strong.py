"""Checks of the stronger learned blind policy before it is used anywhere (no paper file is touched).
1. The hand-written rule family of the paper (code_snapshot/blind_baselines.py + blind_duration_envelope.py: time rules
   (A, B, r) and duration-prior rules (c, B, r)) re-implemented in the same per-step scorer; its single-rule hull must
   reproduce the vertices of gpu2/results/blind_envelope_*_test.json.
2. Three information sets for the learned policy (same models, same policy grid as blind_strong.py):
     full   : as blind_strong.py (adds the observed durations of the earlier steps of the recording and the plan position)
     nohist : without anything from earlier steps of the recording (speed, last ratio, time since the recording began)
     strict : the paper's input types only -- instruction text, step index, step start / elapsed time, fitted priors
   The trees and their priors use the combined QICD train and validation data, as in blind_strong.py.
3. Every vertex of every hull used below is re-scored with the official eval.py (local_eval.py); abort on mismatch.
4. Selection optimism: 500 random half splits of the test recordings. The candidate threshold grid uses model scores on
   the full test set without labels. Policies on the upper hull of half A are then scored on half B, together with ECT.
   We also report the in-sample comparison against the frontier tuned directly on B.
python verify_strong.py OUT_DIR"""
import sys, os, json, math, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.abspath(sys.argv[1])
sys.argv = [sys.argv[0], OUT]; exec(open(os.path.join(HERE, "blind_strong.py")).read().split("DS = {")[0])
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC  # noqa: E402
B_ = "DATA_ROOT/"
MAIN = os.environ.get("QIC_GT_DUMP", "").startswith("gt_dump_main")   # QICD Main set (109 test videos): no Ego-MC
KEY = {"qicd": ("neurips2025", "test", "neurips2025_test")} if MAIN else {"qicd": ("neurips2025", "test", "neurips2025_test"), "mc": ("ego_mc_bench", "test", "ego_mc_bench_test")}

# ---------------- policies as per-step desired (confirm time, alarm time or None)
TRD = collections.defaultdict(list); TRM = collections.defaultdict(lambda: [0, 0])
for v in TR:
    for s in TR[v]: TRD[norm(s["ins"])].append(s["s1"] - s["s0"])
for a in DUMP["neurips2025_train"]:   # train mistake rate exactly as blind_baselines.py (from the output stream)
    cur = None
    for x, y in zip(a["output_texts"], a["output_types"]):
        if y == "instruction": cur = norm(x); TRM[cur][1] += 1
        elif "mistake" in y and cur is not None and TRM[cur][0] < TRM[cur][1]: TRM[cur][0] += 1
GMED = st.median([d for L_ in TRD.values() for d in L_])
med = lambda s: st.median(TRD[norm(s)]) if TRD.get(norm(s)) else GMED
# the paper's rules used the raw HF train rows (blind_baselines.py); that table is exported verbatim (inputs/mrate_1118.json)
TRM = collections.defaultdict(lambda: [0, 0], json.load(open(os.path.join(HERE, "inputs", os.environ.get("QIC_MRATE", "mrate_1118.json")))))
mrate = lambda s: (TRM[norm(s)][0] / TRM[norm(s)][1]) if TRM[norm(s)][1] else None
def rule_grid():
    for c in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0]:
        for B in [None, 15, 25]:
            for r in ([None] if B is None else [None, 0.3]): yield ("dur", c, B, r)
    for A in range(10, 181, 10):
        for B in [None] + list(range(5, 181, 10)):
            if B is not None and B >= A: continue
            for r in ([None] if B is None else [None, 0.2, 0.4, 0.6]): yield ("time", A, B, r)
def rule_wants(V, pol):
    kind, A, B, r = pol; W_ = {}
    for v, S in V.items():
        W_[v] = [(s["s0"] + (A * med(s["ins"]) if kind == "dur" else A),
                  (s["s0"] + B) if (B is not None and (r is None or (mrate(s["ins"]) or 0) >= r)) else None) for s in S]
    return W_
COLS = {"full": list(range(X.shape[1])), "nohist": [c for c in range(X.shape[1]) if c not in (8, 9, 10)],
        "strict": [c for c in range(X.shape[1]) if c not in (6, 7, 8, 9, 10)]}
MODELS = {}
for name, cols in COLS.items():
    MODELS[name] = (HR(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[:, cols], yd),
                    HC(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[:, cols], ym), cols)
def learned_pred(V, name):
    Dm_, Mm_, cols = MODELS[name]; out = {}
    for v, S in V.items():
        x = video_feats(S, PF)[:, cols]; out[v] = (np.exp(Dm_.predict(x)), Mm_.predict_proba(x)[:, 1])
    return out
def learned_grid(V, PRd):
    allpm = np.concatenate([PRd[v][1] for v in V])
    taus = [None] + sorted({float(np.quantile(allpm, q)) for q in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)})
    return [(m, o, a, b, t) for m in MS for o in OS for t in taus for a, b in ([(0, 0)] if t is None else [(a, b) for a in AS for b in BS])]
MS = (.7, .8, .9, 1.0, 1.1, 1.2, 1.4); OS = (-10, -5, 0, 5, 10); AS = (0.0, .25, .5, .75, .9, 1.0, 1.1); BS = (-10, -5, 0, 5, 10, 15, 25)
def learned_wants(V, PRd, pol):
    m, o, a, b, tau = pol
    return {v: [(s["s0"] + m * PRd[v][0][k] + o, (s["s0"] + a * PRd[v][0][k] + b) if (tau is not None and PRd[v][1][k] >= tau) else None)
                for k, s in enumerate(S)] for v, S in V.items()}

# ---------------- per-video scoring (official logic; checked against eval.py below)
def pv_counts(V, wants):
    """dict video -> (ic_hits, clean_steps, tp, fp, fn)"""
    out = {}
    for v, S in V.items():
        end_all = max(s["s1"] for s in S) + 120; ic = cl = tp = fp = fn = 0
        for k, s in enumerate(S):
            wc, wa = wants[v][k]; tc = grid_time(s["s0"], wc, end_all); ta = grid_time(s["s0"], wa, end_all) if wa is not None else None
            if ta is not None and tc is not None and ta >= tc: ta = None
            cl += not s["hm"]
            if not s["ok"]: fn += s["nmis"] if s["hm"] else 0; continue
            if not s["hm"]:
                if ta is not None: fp += 1
                elif tc is not None and abs(tc - s["s1"]) < W: ic += 1
                continue
            if ta is None: fn += s["nmis"]; continue
            j = min(range(len(s["mis"])), key=lambda i: abs(s["mis"][i] - ta)) if s["mis"] else None
            if j is not None and abs(s["mis"][j] - ta) < W: tp += 1; fn += s["nmis"] - 1
            else: fp += 1; fn += s["nmis"]
        out[v] = (ic, cl, tp, fp, fn)
    return out
def wants_to_preds(V, wants):
    P = []
    for v, S in V.items():
        end_all = max(s["s1"] for s in S) + 120; texts, ts = [], []
        for k, s in enumerate(S):
            wc, wa = wants[v][k]; tc = grid_time(s["s0"], wc, end_all); ta = grid_time(s["s0"], wa, end_all) if wa is not None else None
            texts.append("Instruction: " + s["ins"]); ts.append(s["s0"])
            if ta is not None and (tc is None or ta < tc): texts.append("Feedback: you made a mistake."); ts.append(ta)
            if tc is not None: texts.append("Success: Good job!"); ts.append(tc)
        P.append({"video_id": v, "pred_texts": texts, "pred_timestamps": ts})
    return P
def agg(c):  # c: array (..., 5) summed over videos
    ic, cl, tp, fp, fn = c; p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return round(100 * ic / max(cl, 1), 1), round(2 * p * r / max(p + r, 1e-9), 4)
def hull_idx(pts):
    """indices of the upper-hull vertices (same construction as frontier_canon / blind_duration_envelope)."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1])); h = []
    for i in order:
        while len(h) >= 2 and (pts[h[-1]][0] - pts[h[-2]][0]) * (pts[i][1] - pts[h[-2]][1]) - (pts[h[-1]][1] - pts[h[-2]][1]) * (pts[i][0] - pts[h[-2]][0]) >= 0: h.pop()
        h.append(i)
    return h
def fr(h, ic):
    best = 0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in h if i >= ic])

def vagg(C):  # C: (P, 5) -> (P, 2) of (IC %, F1), same rounding as agg
    ic, cl, tp, fp, fn = C.T.astype(float); p = tp / np.maximum(tp + fp, 1); r = tp / np.maximum(tp + fn, 1)
    return np.stack([np.round(100 * ic / np.maximum(cl, 1), 1), np.round(2 * p * r / np.maximum(p + r, 1e-9), 4)], 1)
def pareto_hull(P2):
    """indices of the upper-hull vertices of the (IC, F1) points; hull vertices are Pareto points, so filter first."""
    o = np.lexsort((-P2[:, 1], -P2[:, 0])); best = -1.0; keep = []
    for i in o:
        if P2[i, 1] > best: keep.append(i); best = P2[i, 1]
    keep = np.array(keep); hv = hull_idx([tuple(P2[i]) for i in keep]); return keep[hv]

res = {"rule_reproduction": {}, "vertex_checks": {}, "families": {}, "crossfit": {}}
FAM = {}
for ds, (dv, sp, key) in KEY.items():
    V = SPL[key]; vids = sorted(V); dso = L.load(dv, sp); fam = {}
    rp = list(rule_grid()); fam["rules"] = (rp, lambda p: rule_wants(V, p))   # wants are rebuilt on demand (memory)
    for name in COLS:
        PRd = learned_pred(V, name); lp = learned_grid(V, PRd); fam[name] = (lp, lambda p, PRd=PRd: learned_wants(V, PRd, p))
    arr = {}
    for f, (pols, mk) in fam.items():
        A_ = np.zeros((len(pols), len(vids), 5), dtype=np.int32)
        for i, pol in enumerate(pols):
            c = pv_counts(V, mk(pol)); A_[i] = [c[v] for v in vids]
        arr[f] = A_
        pts = [agg(A_[i].sum(0)) for i in range(len(pols))]; hv = hull_idx(pts)
        for i in hv:  # official check of every vertex
            off, _ = L.run(dso, wants_to_preds(V, mk(pols[i])))
            if (off["ic"], round(off["f1"], 4)) != pts[i]: sys.exit(f"VERTEX MISMATCH {ds} {f} {pols[i]} {pts[i]} {off}")
        res["vertex_checks"][f"{ds}/{f}"] = len(hv)
        res["families"][f"{ds}/{f}"] = dict(n=len(pols), hull=[(pts[i][0], pts[i][1], str(pols[i])) for i in hv],
                                            f1_max=max(p[1] for p in pts), ic_max=max(p[0] for p in pts))
        print(ds, f, "policies", len(pols), "hull", [(pts[i][0], round(pts[i][1], 3)) for i in hv], flush=True)
    old = json.load(open(B_ + ("machine4/results/blind_envelope_neurips2025_main_test.json" if MAIN else f"machine0/results/blind_envelope_{dv}_test.json")))["hull_vertices"]
    mine = [(p[0], round(p[1], 3)) for p in (r[:2] for r in res["families"][f"{ds}/rules"]["hull"])]
    res["rule_reproduction"][ds] = dict(old=[(h[0], h[1]) for h in old], mine=mine)
    print(ds, "rule hull old", [(h[0], h[1]) for h in old], "\n   mine", mine, flush=True)
    om = {(round(a, 1), round(b, 3)) for a, b, *_ in old}; mm = {(round(float(a), 1), round(float(b), 3)) for a, b in mine}
    if not om <= {(a, b) for a, b in mm} | {(a, round(b + d, 3)) for a, b in mm for d in (-.001, .001)}: sys.exit(f"RULE HULL DOES NOT REPRODUCE {ds}")
    FAM[ds] = (vids, arr)

# ---------------- ECT runs, per video, official eval per recording
RUNS = {"qicd": {"greedy": ["results/final/sft_greedy_test36", "machine4/results/seeds/final/s1_greedy_test36", "machine5/results/seeds/final/s2_greedy_test36", "machine7/results/jobs/s3_greedy_test36"],
                 "thr": ["results/final/sft_prob_test36", "machine4/results/seeds/final/s1_prob_test36", "machine5/results/seeds/final/s2_prob_test36", "machine6/results/seeds/final/s3_prob_test36"]},
        "mc": {"greedy": ["machine0/results/final/sft_greedy_mc", "machine4/results/seeds/final/s1_greedy_mc", "machine5/results/seeds/final/s2_greedy_mc", "machine5/results/jobs/s3_greedy_mc"],
               "thr": ["machine0/results/final/sft_prob_mc", "machine4/results/seeds/final/s1_prob_mc", "machine5/results/seeds/final/s2_prob_mc", "machine5/results/jobs/s3_prob_mc"]}}
if MAIN:
    RUNS = {"qicd": {"greedy": [f"machine8/results/jobs/MAIN_ECT_s{i}" for i in range(4)],
                     "thr": ["machine8/results/jobs/MAIN_ECTthr_s0", "machine8/results/jobs/MAIN_ECTthr_s1", "machine5/results/jobs/MAIN_ECTthr_s2", "machine8/results/jobs/MAIN_ECTthr_s3"]}}
ECT = {}
for ds, (dv, sp, key) in KEY.items():
    dso = L.load(dv, sp); vids = FAM[ds][0]; cl = {v: sum(not s["hm"] for s in SPL[key][v]) for v in vids}
    for kind, paths in RUNS[ds].items():
        for si, p in enumerate(paths):
            P = {x["video_id"]: x for x in json.load(open(B_ + p + "/pred.json"))}; a = np.zeros((len(vids), 5), dtype=np.int32)
            v0 = next(v for v in vids if cl[v] > 0); r0, _ = L.run(dso, [P[v0]])
            for j, v in enumerate(vids):
                if cl[v] > 0: r, _ = L.run(dso, [P[v]]); a[j] = (round(r["ic"] * cl[v] / 100), cl[v], r["tp"], r["fp"], r["fn"])
                else:  # eval.py divides by zero on a recording without mistake-free steps: score it together with v0, subtract v0
                    r, _ = L.run(dso, [P[v0], P[v]]); a[j] = (0, 0, r["tp"] - r0["tp"], r["fp"] - r0["fp"], r["fn"] - r0["fn"])
            tot, _ = L.run(dso, list(P.values())); assert agg(a.sum(0)) == (tot["ic"], round(tot["f1"], 4)), (p, agg(a.sum(0)), tot)
            ECT[(ds, kind, si)] = a
print("ECT per-video counts reproduce the official totals", flush=True)

# ---------------- in-sample margins on the full test set and the split-half cross-fit
rng = np.random.default_rng(0); NS = 500
for ds in KEY:
    vids, arr = FAM[ds]; n = len(vids); fams = {"rules": arr["rules"], "strict": np.concatenate([arr["rules"], arr["strict"]]),
                                               "nohist": np.concatenate([arr["rules"], arr["nohist"]]), "full": np.concatenate([arr["rules"], arr["full"]])}
    out = {}
    for fname, A_ in fams.items():
        pa_ = vagg(A_.sum(1)); H_all = [tuple(pa_[i]) for i in pareto_hull(pa_)]
        full_m = {f"{k}_s{s}": round(agg(ECT[(ds, k, s)].sum(0))[1] - fr(H_all, agg(ECT[(ds, k, s)].sum(0))[0]), 3) for k in ("greedy", "thr") for s in range(4)}
        oos, ins = collections.defaultdict(list), collections.defaultdict(list)
        for _ in range(NS):
            perm = rng.permutation(n); a_, b_ = perm[: n // 2], perm[n // 2:]
            pa = vagg(A_[:, a_, :].sum(1)); ha = pareto_hull(pa)          # policies on the hull of half A
            pbA = vagg(A_[ha][:, b_, :].sum(1)); Hb_oos = [tuple(pbA[i]) for i in hull_idx([tuple(x) for x in pbA])]
            pb = vagg(A_[:, b_, :].sum(1)); hb = pareto_hull(pb); Hb_in = [tuple(pb[i]) for i in hb]
            for k in ("greedy", "thr"):
                for s in range(4):
                    e = agg(ECT[(ds, k, s)][b_].sum(0)); oos[f"{k}_s{s}"].append(e[1] - fr(Hb_oos, e[0])); ins[f"{k}_s{s}"].append(e[1] - fr(Hb_in, e[0]))
        summ = {r: dict(margin_full_test=full_m[r], oos_mean=round(float(np.mean(oos[r])), 3), oos_p_beyond=round(float(np.mean(np.array(oos[r]) > 0)), 3),
                        ins_mean=round(float(np.mean(ins[r])), 3), ins_p_beyond=round(float(np.mean(np.array(ins[r]) > 0)), 3)) for r in oos}
        out[fname] = dict(n_policies=len(A_), frontier_full_test=H_all, ect=summ)
        print(f"\n{ds} family={fname} ({len(A_)} policies)  frontier on full test:", [(round(a, 1), round(b, 3)) for a, b in H_all])
        for r, d in summ.items(): print(f"  {r:9s} margin(full test, tuned) {d['margin_full_test']:+.3f} | half-split: in-sample {d['ins_mean']:+.3f} P>0 {d['ins_p_beyond']:.2f} | out-of-sample {d['oos_mean']:+.3f} P>0 {d['oos_p_beyond']:.2f}", flush=True)
    res["crossfit"][ds] = out
json.dump(res, open(os.path.join(OUT, "verify_strong_main.json" if MAIN else "verify_strong.json"), "w"), indent=1, default=str)
print("VERIFY_DONE")
