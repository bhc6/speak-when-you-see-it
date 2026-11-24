"""Recipe-grouped 5-fold CV over a policy_search table; bootstrap CIs; paired difference vs B0.
python cv_select.py SEARCH_JSON"""
import json, sys
import numpy as np

S = json.load(open(sys.argv[1])); folds, T = S["folds"], S["table"]
vids = sorted(folds); K = 5
B0, B1, B2 = "text|text|1000000000", "text|text|1", "text|never|1000000000"


def tot(cfg, vs): return np.array([T[cfg][v] for v in vs]).sum(0)


def met(c):
    ic, den, tp, fp, tn, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"IC": 100 * ic / den, "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "FP": fp, "TP": tp}


def obj(c, o): m = met(c); return m[o] + (1e-6 * m["IC"] if o == "F1" else 1e-6 * m["F1"])


rng = np.random.default_rng(0)
def boot(rows, R=10000):
    rows = np.array(rows); idx = rng.integers(0, len(rows), (R, len(rows))); s = rows[idx].sum(1)
    ic, den, tp, fp, tn, fn = np.moveaxis(s, -1, 0)
    with np.errstate(all="ignore"):
        p = tp / (tp + fp); r = tp / (tp + fn); f1 = np.nan_to_num(2 * p * r / (p + r))
    return {"IC": 100 * ic / den, "F1": f1, "P": np.nan_to_num(p), "R": r, "FP": fp}


def fmt(m): return f"IC {m['IC']:.1f}  P {m['P']:.3f}  R {m['R']:.3f}  F1 {m['F1']:.3f}  TP {m['TP']}  FP {m['FP']}"


print("in-sample (all 57):")
for name, c in (("B0 baseline rule", B0), ("B1 first alert only", B1), ("B2 no alerts", B2)):
    print(f"  {name:22s} {fmt(met(tot(c, vids)))}")
for o in ("F1", "IC"):
    best = max(T, key=lambda c: obj(tot(c, vids), o))
    print(f"  best-in-sample[{o}] {best:28s} {fmt(met(tot(best, vids)))}   (optimistic, not for reporting)")
for o in ("F1", "IC"):
    oof_rows, chosen = {}, []
    for k in range(K):
        tr = [v for v in vids if folds[v] != k]; te = [v for v in vids if folds[v] == k]
        best = max(T, key=lambda c: obj(tot(c, tr), o)); chosen.append(best)
        for v in te: oof_rows[v] = T[best][v]
    rows = [oof_rows[v] for v in vids]; base = [T[B0][v] for v in vids]
    m = met(np.array(rows).sum(0)); b = boot(rows); bb = boot(base)
    rng2 = np.random.default_rng(1); idx = rng2.integers(0, len(vids), (10000, len(vids)))
    A, Bm = np.array(rows)[idx].sum(1), np.array(base)[idx].sum(1)
    def m_arr(s):
        ic, den, tp, fp, tn, fn = np.moveaxis(s, -1, 0)
        with np.errstate(all="ignore"):
            p = tp / (tp + fp); r = tp / (tp + fn)
        return 100 * ic / den, np.nan_to_num(2 * p * r / (p + r))
    (ia, fa), (ib, fb) = m_arr(A), m_arr(Bm)
    print(f"\nCV objective {o}: chosen per fold {chosen}")
    print(f"  out-of-fold   {fmt(m)}")
    print(f"     95% CI IC [{np.percentile(b['IC'],2.5):.1f}, {np.percentile(b['IC'],97.5):.1f}]  F1 [{np.percentile(b['F1'],2.5):.3f}, {np.percentile(b['F1'],97.5):.3f}]")
    print(f"  paired vs B0: dIC {np.mean(ia-ib):+.1f} [{np.percentile(ia-ib,2.5):+.1f}, {np.percentile(ia-ib,97.5):+.1f}]  dF1 {np.mean(fa-fb):+.3f} [{np.percentile(fa-fb,2.5):+.3f}, {np.percentile(fa-fb,97.5):+.3f}]")
