"""Same alarm budget for "+ clean" (ABL_noND): exact offline replay of a qic_prob run (order done_first) at a LOWER mistake
threshold than the one it was run with (rethresh.py can only raise it).
Why it is exact: in done_first the completion question is asked first at every check and its decision (logged "yes") never
depends on the mistake decision, so the sequence of checks is fixed; the probability of every mistake call is logged, so the
mistake decision can be re-taken at any threshold. Check j of a step (j = 1, 2, ...) is at t = (int(2*start) + 10 j) / 2; a
step starts in the log where the completion call has n == 10 frames.
Checks before anything is used: (1) at the logged threshold the replay reproduces the original shard predictions exactly
(texts kinds and time stamps); (2) the official evaluation of the replay equals the run's eval.log.
Budget rule as pre-declared for supp Table "same alarm budget" (EXPERIMENTS 11-12 00:10): the smallest threshold on a 0.01 grid
whose scored alarms on the validation split (official TP + FP) are at most those of frozen thresholded ECT (387); the same rule
must first reproduce the thresholds of the two variants already in the table (0.63 / 370 and 0.51 / 382). Applied once to test.
Comparison with thresholded ECT seed 0: paired bootstrap over videos (2000), 95% interval, TOST with pre-declared bounds
+-5 IC-Acc and +-0.05 F1 (90% interval inside the bounds), as in analysis_code/eq_results_1112.txt.
python budget_clean.py OUT_DIR"""
import sys, os, json, glob, random
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
RUN = {"val": ("neurips2025", "validation", B + "results/jobs/ABL_noND_prob_val"),
       "qicd": ("neurips2025", "test", B + "results/jobs/ABL_noND_prob_test36"),
       "mc": ("ego_mc_bench", "test", B + "machine3/results/jobs/ABL_noND_prob_mc")}
ECT = {"val": B + "results/prob/T1v4_s2400_done_first_0.2_0.35", "qicd": B + "results/final/sft_prob_test36", "mc": B + "machine0/results/final/sft_prob_mc"}

def load_run(d):
    """list of (shard predictions, per-step list of checks [(done_yes, p_mis)])"""
    out = []
    for sj in sorted(glob.glob(f"{d}/shard*.json")):
        k = sj.rsplit("shard", 1)[1][:-5]; S = json.load(open(sj)); Lg = [json.loads(l) for l in open(f"{d}/plog{k}.jsonl")]
        steps, cur, i = [], None, 0
        while i < len(Lg):
            r = Lg[i]; assert r["mode"] == "instruction_end", (sj, i, r)
            if r["n"] == 10 or cur is None: cur = []; steps.append(cur)
            if r["yes"]: cur.append((True, None)); i += 1
            else:
                m = Lg[i + 1]; assert m["mode"] == "mistake_inference" and m["n"] == r["n"], (sj, i, m); cur.append((False, m)); i += 2
        nsteps = sum(t.startswith("Instruction") for p in S for t in p["pred_texts"])
        assert nsteps == len(steps), (sj, nsteps, len(steps))
        out.append((S, steps))
    return out

def replay(run, tau=None):
    """predictions at mistake threshold tau (None: the logged decisions)"""
    P = []
    for S, steps in run:
        it = iter(steps)
        for p in S:
            texts, ts = [], []; orig = list(zip(p["pred_texts"], p["pred_timestamps"])); oi = 0; fbtext = iter([x for x, _ in orig if x.startswith("Feedback")])
            for x, t in orig:
                if not x.startswith("Instruction"): continue
                st = t; texts.append(x); ts.append(t); init = int(st * 2)
                for j, (dy, m) in enumerate(next(it), 1):
                    tj = (init + 10 * j) / 2
                    if dy: texts.append("Success: Good job!"); ts.append(tj); break
                    fire = m["yes"] if tau is None else m["p"] >= tau
                    if m["yes"]: ftxt = next(fbtext)
                    if fire: texts.append(ftxt if m["yes"] else "Feedback: you made a mistake."); ts.append(tj)
            P.append({"video_id": p["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return P

def kinds(P): return [[(x.split(":")[0], round(t, 3)) for x, t in zip(p["pred_texts"], p["pred_timestamps"])] for p in P]
def n_alarms(P): return sum(x.startswith("Feedback") for p in P for x in p["pred_texts"])
def eval_log_last(d):
    import re; o = open(f"{d}/eval.log").read(); ic = float(re.findall(r"IC-Acc: ([\d.]+)", o)[-1])
    return (ic, *map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1]))

res = {}; runs = {}
for k, (dv, sp, d) in RUN.items():
    run = load_run(d); runs[k] = run; orig = [x for sj in sorted(glob.glob(f"{d}/shard*.json")) for x in json.load(open(sj))]
    R = replay(run)
    assert kinds(R) == kinds(orig), f"replay at the logged threshold differs from the shard predictions ({k})"
    off, _ = L.run(L.load(dv, sp), R); el = eval_log_last(d)
    assert (off["ic"], off["tp"], off["fp"], off["tn"], off["fn"]) == el, (k, off, el)
    res[f"check_{k}"] = dict(alarms=n_alarms(R), eval=off)
    print(k, "replay reproduces shard predictions and eval.log:", off, "alarms", n_alarms(R), flush=True)

# budget on validation, the pre-declared rule of EXPERIMENTS 11-12 00:10 (E1): the smallest tau on a 0.01 grid whose number of
# scored alarms on the validation split (official TP + FP) is at most that of frozen thresholded ECT (387)
DSV = L.load("neurips2025", "validation")
def val_alarms(P): r, _ = L.run(DSV, P); return r["tp"] + r["fp"]
ect_val = val_alarms(json.load(open(ECT["val"] + "/pred.json"))); assert ect_val == 387, ect_val
GRID = [round(0.01 * i, 2) for i in range(1, 100)]
# the rule must reproduce the two thresholds of the paper's table before it is applied to "+ clean"
for name, d, want, want_n in (("No paired neg.", B + "results/jobs/ABL_none_prob_val", 0.63, 370), ("+ notdone", B + "machine7/results/jobs/ABL_noCL_prob_val", 0.51, 382)):
    rv = load_run(d); t = next(t for t in GRID if t >= 0.35 - 1e-9 and val_alarms(replay(rv, t)) <= ect_val)
    n = val_alarms(replay(rv, t)); print(name, "tau", t, "val alarms", n, flush=True); assert (t, n) == (want, want_n), (name, t, n)
tau = next(t for t in GRID if val_alarms(replay(runs["val"], t)) <= ect_val)
res["budget"] = dict(ect_val_alarms=ect_val, tau=tau, clean_val_alarms_at_tau=val_alarms(replay(runs["val"], tau)),
                     clean_val_alarms_frozen=val_alarms(replay(runs["val"])), clean_val_eval_at_tau=L.run(DSV, replay(runs["val"], tau))[0])
print("budget", res["budget"], flush=True)

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
    return 100 * ic / max(cl, 1), 2 * p * r / max(p + r, 1e-9), tp + fp
for k in ("qicd", "mc"):
    dv, sp, d = RUN[k]; ds = L.load(dv, sp)
    E = per_video(ds, json.load(open(ECT[k] + "/pred.json")))
    for label, P in (("frozen", replay(runs[k])), ("budget", replay(runs[k], tau))):
        C = per_video(ds, P); V = sorted(E); ea, eb = agg(E, V), agg(C, V)
        tot, _ = L.run(ds, P); assert abs(eb[0] - tot["ic"]) < 0.051 and abs(eb[1] - tot["f1"]) < 1e-3, (k, label, eb, tot)
        rng = random.Random(0); di, df = [], []
        for _ in range(2000):
            S_ = [rng.choice(V) for _ in V]; x, y = agg(E, S_), agg(C, S_); di.append(y[0] - x[0]); df.append(y[1] - x[1])
        di.sort(); df.sort(); q = lambda L_, a: (L_[int(a * 2000)], L_[int((1 - a) * 2000) - 1])
        r = dict(ect=(round(ea[0], 1), round(ea[1], 3), ea[2]), clean=(round(eb[0], 1), round(eb[1], 3), eb[2]),
                 dIC=round(eb[0] - ea[0], 1), dIC95=[round(x, 1) for x in q(di, .025)], dIC90=[round(x, 1) for x in q(di, .05)],
                 dF1=round(eb[1] - ea[1], 3), dF195=[round(x, 3) for x in q(df, .025)], dF190=[round(x, 3) for x in q(df, .05)])
        r["equiv_IC"] = -5 < r["dIC90"][0] and r["dIC90"][1] < 5; r["equiv_F1"] = -.05 < r["dF190"][0] and r["dF190"][1] < .05
        res[f"{k}_{label}"] = r; print(k, label, r, flush=True)
json.dump(res, open(f"{OUT}/budget_clean.json", "w"), indent=1)
print("BUDGET_DONE")
