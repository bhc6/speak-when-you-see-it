"""A stronger learned video-blind policy (supplement B; answers the limitation "a stronger blind learner could still extend it").
CPU only, no video. Run: python blind_strong.py OUT_DIR

What the policy may see (all available to any policy under the official per-step, turn-based protocol, none of it visual):
  instruction text, step index and plan length, the start time of the current step and of every earlier step of this
  recording (the protocol reveals step starts), hence the observed durations of the earlier steps ("how fast this person
  is"), and per-instruction duration and mistake priors from the fitting data.
Learned parts (scikit-learn HistGradientBoosting, with priors out-of-fold by recipe for fitting):
  The official loader combines QICD train and validation under the train key. No test recordings are used to fit the trees.
  D: log step duration (end = last annotated time of the step) -> predicted end  E^ = start + exp(D)
  M: P(step contains an annotated mistake)
  R: median relative mistake time within the step in the fitting data (QICD: annotated near the step end)
Policy family (one decision per step, executed on the official 5-s check grid; Success before Feedback at the same check,
as in blind_learned.py and the hand-written rules):
  confirm ("Success") at start + m * (E^ - start) + o
  alarm ("Feedback") once at start + a * (E^ - start) + b   iff  M >= tau
  grid m x o x a x b x tau (tau as quantiles of M on the scored split, plus 'never').
Scoring: fast per-step re-implementation of the official eval.py logic (IC: clean step, exactly one Success within 15 s
of the step end; TP/FP/FN by the closest annotated mistake within 15 s). Verified exactly against the official eval.py
(local_eval.py) on random policies of every split before results are written; the script aborts on any mismatch.
Selection as for the hand-written rules: favorable = best on the scored test set itself; also validation-selected."""
import sys, os, json, math, random, hashlib, collections, statistics as st
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import local_eval as L  # noqa: E402
OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
DUMP = json.load(open(os.path.join(HERE, "inputs", os.environ.get("QIC_GT_DUMP", "gt_dump_1118.json"))))   # QICD Main: QIC_GT_DUMP=gt_dump_main_1119.json
W = 15.0
def norm(s): return " ".join(s.strip().lower().rstrip(".").split())

def steps(key):
    """Per video: list of steps (instruction, start, end, has_mistake, [(mistake_time)], obs_dur_to_next_start)."""
    V = {}
    for a in DUMP[key]:
        S = []
        for tx, ts, ty, hm in zip(a["instruction_segment_texts"], a["instruction_segment_texts_timestamps"],
                                  a["instruction_segment_texts_types"], a["instrcution_segment_has_mistake"]):
            mis = [t for t, y, x in zip(ts, ty, tx) if "mistake" in y and x is not None]
            # ok: the official eval first checks the instruction by ROUGE-L >= 0.8 (after stripping "Instruct:"); empty GT
            # instructions (end-of-video segments) fail it, and such a step only adds its mistakes as false negatives
            ok = L.ev.check_if_correct_instruction_given(tx[0], ("Instruction: " + tx[0]).replace("Instruct:", "").strip())
            S.append(dict(ins=tx[0], s0=ts[0], s1=max(ts), hm=bool(hm), mis=mis, nmis=sum("mistake" in y for y in ty), ok=ok, ts=ts, ty=ty, tx=tx))
        V[a["video_id"]] = S
    return V
SPL = {k: steps(k) for k in ("neurips2025_train", "neurips2025_validation", "neurips2025_test", "ego_mc_bench_test") if k in DUMP}
TR = SPL["neurips2025_train"]; recipe = lambda v: v.split("_")[0]

def priors(videos):
    d = collections.defaultdict(list); m = collections.defaultdict(lambda: [0, 0]); rel = []
    for v in videos:
        for s in TR[v]:
            dur = max(s["s1"] - s["s0"], 1.0); d[norm(s["ins"])].append(dur); m[norm(s["ins"])][0] += s["hm"]; m[norm(s["ins"])][1] += 1
            rel += [(t - s["s0"]) / dur for t in s["mis"]]
    g = st.median([x for L_ in d.values() for x in L_]); gm = sum(x[0] for x in m.values()) / max(sum(x[1] for x in m.values()), 1)
    def P(ins):
        k = norm(ins); x = d.get(k)
        if not x: return g, g * .6, g * 1.6, 0, gm
        q = np.quantile(x, [.25, .75]); return st.median(x), q[0], q[1], len(x), m[k][0] / m[k][1]
    return P, (st.median(rel) if rel else 1.0)

H = 64
def hvec(ins):
    v = np.zeros(H); w = norm(ins).split()
    for t in w + [a + "_" + c for a, c in zip(w, w[1:])]: v[int(hashlib.md5(t.encode()).hexdigest(), 16) % H] += 1
    return v / max(np.linalg.norm(v), 1)

def video_feats(S, P):
    """Features of every step of one recording, using only what is known when the step starts."""
    X = []; logr = []
    for k, s in enumerate(S):
        med, q1, q3, n, pm = P(s["ins"])
        sp = math.exp(np.mean(logr)) if logr else 1.0; last = math.exp(logr[-1]) if logr else 1.0
        X.append(np.concatenate([[math.log(med), math.log(max(q1, 1)), math.log(max(q3, 1)), min(n, 50) / 50, pm,
                                  min(k, 40) / 40, len(S) / 40, k / max(len(S) - 1, 1), math.log(sp), math.log(last),
                                  (s["s0"] - S[0]["s0"]) / 600], hvec(s["ins"])]))
        if k + 1 < len(S):  # the protocol reveals the next start, so this step's observed duration is known afterwards
            obs = max(S[k + 1]["s0"] - s["s0"], 1.0); logr.append(math.log(obs / med))
    return np.array(X)

# ---- fit on train, priors out-of-fold by recipe (5 folds) so train features never see their own recording
vids = sorted(TR); recs = sorted({recipe(v) for v in vids}); fold = {r: i % 5 for i, r in enumerate(recs)}
X, yd, ym = [], [], []
for f in range(5):
    P, _ = priors([v for v in vids if fold[recipe(v)] != f])
    for v in vids:
        if fold[recipe(v)] != f: continue
        X.append(video_feats(TR[v], P)); yd += [math.log(max(s["s1"] - s["s0"], 1.0)) for s in TR[v]]; ym += [s["hm"] for s in TR[v]]
X = np.vstack(X); yd = np.array(yd); ym = np.array(ym, dtype=int)
print("train steps", len(X), "mistake rate", round(ym.mean(), 3), flush=True)
Dm = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, yd)
Mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, ym)
PF, REL = priors(vids)
print("median relative mistake time on train", round(REL, 3), flush=True)

def predict(V):
    out = {}
    for v, S in V.items():
        x = video_feats(S, PF); out[v] = (np.exp(Dm.predict(x)), Mm.predict_proba(x)[:, 1])
    return out

def grid_time(s0, want, end_all):
    """First official check time >= want (checks at (int(2*s0) + 10j)/2, j >= 1); None if past the end of the video."""
    init = int(s0 * 2); j = max(1, math.ceil((want * 2 - init) / 10 - 1e-9)); t = (init + 10 * j) / 2
    return None if t > end_all else t

def score(V, PR, pol):
    m, o, a, b, tau = pol; ic = cl = tp = fp = fn = 0
    for v, S in V.items():
        dur, pm = PR[v]; end_all = max(s["s1"] for s in S) + 120
        for k, s in enumerate(S):
            s0 = s["s0"]; tc = grid_time(s0, s0 + m * dur[k] + o, end_all)
            ta = grid_time(s0, s0 + a * dur[k] + b, end_all) if (tau is not None and pm[k] >= tau) else None
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
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(ic=round(100 * ic / max(cl, 1), 1), tp=tp, fp=fp, fn=fn, f1=round(2 * p * r / max(p + r, 1e-9), 4))

def to_preds(V, PR, pol):
    """The same policy as official prediction lists, for the exact check against eval.py."""
    m, o, a, b, tau = pol; P = []
    for v, S in V.items():
        dur, pm = PR[v]; end_all = max(s["s1"] for s in S) + 120; texts, ts = [], []
        for k, s in enumerate(S):
            s0 = s["s0"]; texts.append("Instruction: " + s["ins"]); ts.append(s0)
            tc = grid_time(s0, s0 + m * dur[k] + o, end_all)
            ta = grid_time(s0, s0 + a * dur[k] + b, end_all) if (tau is not None and pm[k] >= tau) else None
            if ta is not None and (tc is None or ta < tc): texts.append("Feedback: you made a mistake."); ts.append(ta)
            if tc is not None: texts.append("Success: Good job!"); ts.append(tc)
        P.append({"video_id": v, "pred_texts": texts, "pred_timestamps": ts})
    return P

DS = {"neurips2025_validation": ("neurips2025", "validation"), "neurips2025_test": ("neurips2025", "test"), "ego_mc_bench_test": ("ego_mc_bench", "test")}
MS = (.7, .8, .9, 1.0, 1.1, 1.2, 1.4); OS = (-10, -5, 0, 5, 10)
AS = (0.0, .25, .5, .75, .9, 1.0, 1.1); BS = (-10, -5, 0, 5, 10, 15, 25)
res = {"train_steps": len(X), "rel_mistake_time_train": REL, "splits": {}}
for key, (dv, sp) in DS.items():
    V = SPL[key]; PR = predict(V); allpm = np.concatenate([PR[v][1] for v in V])
    taus = [None] + sorted({float(np.quantile(allpm, q)) for q in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)})
    pols = [(m, o, a, b, t) for m in MS for o in OS for t in taus for a, b in ([(0, 0)] if t is None else [(a, b) for a in AS for b in BS])]
    # exact check against the official eval.py on 40 random policies (plus the two extremes)
    rnd = random.Random(0); chk = rnd.sample(pols, 40) + [pols[0], pols[-1]]; ds = L.load(dv, sp)
    for pol in chk:
        mine = score(V, PR, pol); off, _ = L.run(ds, to_preds(V, PR, pol))
        if (mine["ic"], mine["tp"], mine["fp"], mine["fn"]) != (off["ic"], off["tp"], off["fp"], off["fn"]):
            sys.exit(f"MISMATCH {key} {pol} fast={mine} official={off}")
    R = []
    for pol in pols: r = score(V, PR, pol); r["pol"] = pol; R.append(r)
    res["splits"][key] = dict(n_policies=len(R), checked_against_official=len(chk), points=R)
    dur_err = np.median([abs(math.log(max(s["s1"] - s["s0"], 1)) - math.log(PR[v][0][k])) for v, S in V.items() for k, s in enumerate(S)])
    best = max(R, key=lambda r: (r["f1"], r["ic"])); bic = max(R, key=lambda r: (r["ic"], r["f1"]))
    res["splits"][key].update(median_abs_log_dur_err=float(dur_err), f1_opt=best, ic_opt=bic)
    print(key, "policies", len(R), "verified", len(chk), "| F1-opt", best["ic"], best["f1"], best["pol"], "| IC-opt", bic["ic"], bic["f1"], bic["pol"],
          "| median |log dur err|", round(float(dur_err), 3), flush=True)
json.dump(res, open(os.path.join(OUT, "blind_strong.json"), "w"))
print("STRONG_DONE")
