"""Same alarm budget, all three variants, with ONE alarm counter for ECT and the variants (correction of the table of the
supplement, whose rule compared the variants' Feedback entries with ECT's scored alarms). Two consistent counters:
  entries: Feedback entries in the predictions (the rule's "number of feedbacks"; ECT frozen: 432 on validation)
  scored : official TP + FP on validation (ECT frozen: 387)
For each counter and variant: the smallest tau on a 0.01 grid with count <= ECT's, replayed exactly (budget_clean.py), applied
once to QICD test and Ego-MC-Bench; paired bootstrap over videos against thresholded ECT seed 0 (2000; 95% CI; TOST +-5 / +-0.05
on the 90% CI). Also the original (inconsistent) rule for reference. python budget_all.py OUT_DIR"""
import sys, os, json, glob, random
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "budget_clean.py")).read().split("res = {}; runs = {}")[0]; OUTD = sys.argv[1]; exec(src)
OUT = os.path.abspath(OUTD)
VAR = {"No paired neg.": ("results/jobs/ABL_none_prob_val", "results/jobs/ABL_none_prob_test36", "machine4/results/jobs/ABL_none_prob_mc"),
       "+ notdone": ("machine7/results/jobs/ABL_noCL_prob_val", "machine7/results/jobs/ABL_noCL_prob_test36", "machine8/results/jobs/ABL_noCL_prob_mc"),
       "+ clean": ("results/jobs/ABL_noND_prob_val", "results/jobs/ABL_noND_prob_test36", "machine3/results/jobs/ABL_noND_prob_mc")}
DS = {"val": L.load("neurips2025", "validation"), "qicd": L.load("neurips2025", "test"), "mc": L.load("ego_mc_bench", "test")}
def scored(P): r, _ = L.run(DS["val"], P); return r["tp"] + r["fp"]
CNT = {"entries": n_alarms, "scored": scored}
ECTP = {k: json.load(open(ECT[k] + "/pred.json")) for k in ECT}
EB = {c: CNT[c](ECTP["val"]) for c in CNT}; print("ECT validation:", EB, flush=True)
GRID = [round(0.01 * i, 2) for i in range(1, 100)]
def per_video(ds, P):
    P = {p["video_id"]: p for p in P}; vids = sorted(P); out = {}
    cl = {v: sum(not x for x in L.ev.get_ground_truth_data(ds, v)["has_mistake"]) for v in vids}
    v0 = next(v for v in vids if cl[v]); r0, _ = L.run(ds, [P[v0]])
    for v in vids:
        if cl[v]: r, _ = L.run(ds, [P[v]]); out[v] = (round(r["ic"] * cl[v] / 100), cl[v], r["tp"], r["fp"], r["fn"])
        else: r, _ = L.run(ds, [P[v0], P[v]]); out[v] = (0, 0, r["tp"] - r0["tp"], r["fp"] - r0["fp"], r["fn"] - r0["fn"])
    return out
def agg(c, V):
    ic, cl, tp, fp, fn = (sum(c[v][i] for v in V) for i in range(5)); p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 100 * ic / max(cl, 1), 2 * p * r / max(p + r, 1e-9)
EPV = {k: per_video(DS[k], ECTP[k]) for k in ("qicd", "mc")}
def compare(k, P):
    E, C = EPV[k], per_video(DS[k], P); V = sorted(E); ea, eb = agg(E, V), agg(C, V)
    rng = random.Random(0); di, df = [], []
    for _ in range(2000):
        S_ = [rng.choice(V) for _ in V]; x, y = agg(E, S_), agg(C, S_); di.append(y[0] - x[0]); df.append(y[1] - x[1])
    di.sort(); df.sort(); q = lambda L_, a: (round(L_[int(a * 2000)], 3), round(L_[int((1 - a) * 2000) - 1], 3))
    r = dict(variant=(round(eb[0], 1), round(eb[1], 3)), ect=(round(ea[0], 1), round(ea[1], 3)), dIC=round(eb[0] - ea[0], 1), dIC95=q(di, .025), dIC90=q(di, .05),
             dF1=round(eb[1] - ea[1], 3), dF195=q(df, .025), dF190=q(df, .05))
    r["equiv_IC"] = -5 < r["dIC90"][0] and r["dIC90"][1] < 5; r["equiv_F1"] = -.05 < r["dF190"][0] and r["dF190"][1] < .05
    return r
res = {"ect_val": EB, "rows": []}
for name, (dv, dt, dm) in VAR.items():
    rv, rt, rm = load_run(B + dv), load_run(B + dt), load_run(B + dm)
    rules = {c: (EB[c], CNT[c]) for c in CNT}
    if name != "+ clean": rules["original"] = (387, n_alarms)   # what the supplement table did (variants' entries vs 387)
    for rule, (budget, f) in rules.items():
        tau = next(t for t in GRID if f(replay(rv, t)) <= budget)
        row = dict(variant=name, rule=rule, tau=tau, val_count=f(replay(rv, tau)), budget=budget,
                   qicd=compare("qicd", replay(rt, tau)), mc=compare("mc", replay(rm, tau)))
        res["rows"].append(row)
        print(f"{name:15s} {rule:8s} tau {tau:.2f} val {row['val_count']}/{budget} | QICD {row['qicd']['variant']} dIC {row['qicd']['dIC']:+.1f} {row['qicd']['dIC95']} dF1 {row['qicd']['dF1']:+.3f} {row['qicd']['dF195']} eq {row['qicd']['equiv_IC']},{row['qicd']['equiv_F1']} | MC {row['mc']['variant']} dIC {row['mc']['dIC']:+.1f} {row['mc']['dIC95']} dF1 {row['mc']['dF1']:+.3f} {row['mc']['dF195']} eq {row['mc']['equiv_IC']},{row['mc']['equiv_F1']}", flush=True)
json.dump(res, open(f"{OUT}/budget_all.json", "w"), indent=1); print("ALL_DONE")
