"""(1) LEARNED video-blind policy + (2) per-video decomposition of every blind policy for bootstrap (CPU only).
Features at each 5-s check (official per-step grid): elapsed time (one-hot 5-s bins up to 180 s + log), step index,
instruction prior median duration and mistake rate (train; OUT-OF-FOLD by recipe on train itself), elapsed/prior,
hashed instruction uni/bigrams (2048). Two logistic regressions (torch, L2) trained on TRAIN checks only:
completion-positive = mistake-free step and |t - end| < 15; mistake-positive = |t - f| < 15 for a GT mistake f.
Policy: at each check, p_done >= td -> Success (step ends); else p_mis >= tm and no feedback yet in step -> Feedback.
Grid of (td, tm) on val (selection) and test (favourable). Outputs per-video counts for bootstrap.
python blind_learned.py EVAL_DIR OUT_JSON"""
import sys, os, io, re, json, math, contextlib, collections, statistics as st, types, hashlib
_bs = types.ModuleType("bert_score"); _bs.BERTScorer = lambda *a, **k: None; sys.modules["bert_score"] = _bs
EV, OUT = sys.argv[1], os.path.abspath(sys.argv[2]); sys.path.insert(0, EV); os.chdir(EV)
import torch, eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
torch.set_num_threads(16)
def norm(s): return " ".join(s.strip().lower().rstrip(".").split())
def load(dv, sp):
    with contextlib.redirect_stdout(io.StringIO()): return utils.load_dataset(dv, "advanced_planning", sp)
TR = load("neurips2025", "train")
def steps_of(ds):
    out = {}
    for a in ds.annotations:
        g = ev.get_ground_truth_data(ds, a["video_id"]); S = []
        for tx, ts, ty, hm in zip(g["texts"], g["texts_timestamps"], g["texts_types"], g["has_mistake"]):
            S.append((tx[0], ts[0], max(ts), hm, [t for t, y in zip(ts[1:], ty[1:]) if "mistake" in y]))
        out[a["video_id"]] = S
    return out
trS = steps_of(TR); recipe = lambda v: v.split("_")[0]
def priors(videos):
    d = collections.defaultdict(list); m = collections.defaultdict(lambda: [0, 0])
    for v in videos:
        for ins, s0, s1, hm, mis in trS[v]: d[norm(ins)].append(s1 - s0); m[norm(ins)][0] += bool(mis); m[norm(ins)][1] += 1
    g = st.median([x for L in d.values() for x in L]) if d else 43.0
    return (lambda s: st.median(d[norm(s)]) if d.get(norm(s)) else g), (lambda s: m[norm(s)][0] / m[norm(s)][1] if m[norm(s)][1] else 0.25)
H = 2048
def feats(ins, k, e, pdur, prate):
    x = [0.0] * (37 + 6 + H); b = min(int(e // 5), 36); x[b] = 1.0
    x[37] = math.log1p(e) / 5; x[38] = min(k, 30) / 30; x[39] = pdur / 100; x[40] = min(e / max(pdur, 1), 4) / 4; x[41] = prate; x[42] = 1.0
    w = norm(ins).split()
    for t in w + [a + "_" + c for a, c in zip(w, w[1:])]: x[43 + int(hashlib.md5(t.encode()).hexdigest(), 16) % H] += 1.0
    return x
def checks(S, pd, pr):
    for k, (ins, s0, s1, hm, mis) in enumerate(S):
        e = 5.0
        while s0 + e <= s1 + 15:
            t = s0 + e; yield k, ins, s0, e, feats(ins, k, e, pd(ins), pr(ins)), float((not hm) and abs(t - s1) < 15), float(any(abs(t - f) < 15 for f in mis)); e += 5
# train with out-of-fold priors (5 recipe folds) so the train features do not see their own video's statistics
vids = sorted(trS); recs = sorted({recipe(v) for v in vids}); fold = {r: i % 5 for i, r in enumerate(recs)}
X, Yd, Ym = [], [], []
for f in range(5):
    pd, pr = priors([v for v in vids if fold[recipe(v)] != f])
    for v in vids:
        if fold[recipe(v)] != f: continue
        for _, _, _, _, x, yd, ym in checks(trS[v], pd, pr): X.append(x); Yd.append(yd); Ym.append(ym)
X = torch.tensor(X); print("train checks", len(X), "pos done", sum(Yd), "pos mis", sum(Ym), flush=True)
def fit(y):
    y = torch.tensor(y); w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pw = (len(y) - y.sum()) / y.sum(); opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, pos_weight=pw) + 1e-3 * (w ** 2).sum(); l.backward(); return l
    opt.step(cl); return w.detach(), b.detach()
Wd, Wm = fit(Yd), fit(Ym)
PD, PR = priors(vids)   # full-train priors for val/test
def policy_preds(ds, S_all, td, tm):
    out = []
    for a in ds.annotations:
        v = a["video_id"]; S = S_all[v]; end_all = max(s1 for _, _, s1, _, _ in S) + 120; texts, ts = [], []
        for k, (ins, s0, s1, hm, mis) in enumerate(S):
            texts.append("Instruction: " + ins); ts.append(s0); init = int(s0 * 2); j = 1; fb = False
            while True:
                t = (init + 10 * j) / 2; e = t - s0
                if t > end_all: break
                x = torch.tensor(feats(ins, k, e, PD(ins), PR(ins)))
                if torch.sigmoid(x @ Wd[0] + Wd[1]).item() >= td: texts.append("Success: Good job!"); ts.append(t); break
                if not fb and torch.sigmoid(x @ Wm[0] + Wm[1]).item() >= tm: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
        out.append({"video_id": v, "pred_texts": texts, "pred_timestamps": ts})
    return out
def per_video(ds, P):
    cl = {p["video_id"]: sum(not x for x in ev.get_ground_truth_data(ds, p["video_id"])["has_mistake"]) for p in P}
    P = sorted(P, key=lambda p: (cl[p["video_id"]] == 0, p["video_id"])); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(ds, P)
    o = buf.getvalue(); C = [tuple(map(int, m)) for m in re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)]
    I = [float(x) for x in re.findall(r"IC-Acc: ([\d.]+)", o)]; res, prev, pdet, cum = {}, (0, 0, 0, 0), 0, 0
    for p, c, ic in zip(P, C, I):
        v = p["video_id"]; cum += cl[v]; det = round(ic * cum / 100)
        res[v] = [*(x - y for x, y in zip(c, prev)), det - pdet, cl[v]]; prev, pdet = c, det
    return res
def agg(pv, V=None):
    V = V or list(pv); tp, fp, tn, fn, det, cl = (sum(pv[v][i] for v in V) for i in range(6))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return 100 * det / max(cl, 1), 2 * p * r / max(p + r, 1e-9)
res = {}
G = [(td, tm) for td in (.3, .4, .5, .6, .7, .8, .9) for tm in (.5, .6, .7, .8, .9, .95, 1.01)]
for dv, sp in (("neurips2025", "validation"), ("neurips2025", "test"), ("ego_mc_bench", "test")):
    ds = load(dv, sp); S_all = steps_of(ds); R = []
    for td, tm in G:
        pv = per_video(ds, policy_preds(ds, S_all, td, tm)); ic, f1 = agg(pv); R.append({"td": td, "tm": tm, "ic": round(ic, 2), "f1": round(f1, 4), "per_video": pv})
    res[f"{dv}_{sp}"] = R; best = max(R, key=lambda r: (r["f1"], r["ic"])); bic = max(R, key=lambda r: (r["ic"], r["f1"]))
    print(dv, sp, "learned blind: F1-opt", best["td"], best["tm"], best["ic"], best["f1"], "| IC-opt", bic["ic"], bic["f1"], flush=True)
json.dump(res, open(OUT, "w")); print("LEARNED_DONE")
