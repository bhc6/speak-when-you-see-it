"""Single-check AUROC of the clock, as in pool_auroc.py (every official 5 s check from the annotated step start to step end
+ 15 s; completion-positive = mistake-free step and |t - end| < 15; mistake-positive = |t - f| < 15 for an annotated
mistake f), with three scores: raw elapsed time e (the number in supplement B), e relative to the training prior Dp of the
same instruction (median annotated duration in the QICD training videos; global median if unseen), and -|e - Dp|.
Priors come from the 209 training videos only (the gt dump's train key also holds the 57 validation videos; they are
removed), so the validation numbers are not fitted on themselves. Also: AUROC of e within each step (pooled over steps
with both labels), which is 1 by construction on QICD because positives are the last checks of a step.
python3 prior_auroc.py -> out/prior_auroc.json"""
import sys, os, json, types, bisect, statistics as st, collections
_t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V0 + "analysis_code/blind_strong")
import local_eval as L
ev = L.ev
OUT = HERE + "/out"; os.makedirs(OUT, exist_ok=True)


def auc(p, n):
    n = sorted(n); return sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p) / max(len(p) * len(n), 1)


norm = lambda s: s.replace("Instruction:", "").strip()
val_ids = {a["video_id"] for a in L.load("neurips2025", "validation").annotations}
tr = L.load("neurips2025", "train"); pr = collections.defaultdict(list); n_train = 0
for a in tr.annotations:
    if a["video_id"] in val_ids: continue
    n_train += 1; g = ev.get_ground_truth_data(tr, a["video_id"])
    for tx, ts in zip(g["texts"], g["texts_timestamps"]):
        if max(ts) > ts[0]: pr[norm(tx[0])].append(max(ts) - ts[0])
PR = {k: st.median(v) for k, v in pr.items()}; GM = st.median([x for v in pr.values() for x in v])
res = dict(n_train_videos=n_train)
for dv, sp in (("neurips2025", "validation"), ("neurips2025", "test"), ("ego_mc_bench", "test")):
    ds = L.load(dv, sp); S = collections.defaultdict(lambda: ([], [])); M = ([], []); within = []; seen = tot = 0
    for a in ds.annotations:
        g = ev.get_ground_truth_data(ds, a["video_id"])
        for tx, ts, ty, hm in zip(g["texts"], g["texts_timestamps"], g["texts_types"], g["has_mistake"]):
            s0, s1 = ts[0], max(ts); mis = [t for t, y in zip(ts[1:], ty[1:]) if "mistake" in y]
            tot += 1; seen += norm(tx[0]) in PR; dp = PR.get(norm(tx[0]), GM)
            e = 5.0; sp_, sn_ = [], []
            while s0 + e <= s1 + 15:
                t = s0 + e; y = (not hm and abs(t - s1) < 15)
                for k, v in (("elapsed", e), ("elapsed_over_prior", e / dp), ("neg_abs_elapsed_minus_prior", -abs(e - dp))):
                    S[k][0 if y else 1].append(v)
                (sp_ if y else sn_).append(e); M[0 if any(abs(t - f) < 15 for f in mis) else 1].append(e)
                e += 5
            if sp_ and sn_: within.append(auc(sp_, sn_))
    r = dict(completion={k: round(auc(*v), 3) for k, v in S.items()}, mistake_elapsed=round(auc(*M), 3),
             within_step_elapsed=round(st.mean(within), 3), steps_with_prior=round(seen / tot, 3),
             n_pos=len(S["elapsed"][0]), n_neg=len(S["elapsed"][1]))
    res[f"{dv}_{sp}"] = r; print(dv, sp, r)
json.dump(res, open(OUT + "/prior_auroc.json", "w"), indent=1)
