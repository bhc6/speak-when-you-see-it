"""Recompute the same-validation-alarm-cap table and the long-step table from the per-video counts alone,
including the bootstrap intervals (same resampling and seed as the original scripts), and compare with expected_tables.json.
Needs only the Python standard library.  python reproduce_tables.py [DATA_DIR]   (default: the directory of this script)
Prints every table cell, then "ALL MATCH" or the cells that differ; exits non-zero on any difference."""
import csv, json, os, random, sys, collections
D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
EXP = json.load(open(os.path.join(D, "expected_tables.json"))); got = {}

# ---- same validation alarm cap
B = collections.defaultdict(dict); TAU = {}
for r in csv.DictReader(open(os.path.join(D, "budget_per_video.csv"))):
    B[(r["benchmark"], r["run"])][r["video"]] = tuple(int(r[k]) for k in ("ic_hits", "mistake_free_steps", "tp", "fp", "fn")); TAU[(r["benchmark"], r["run"])] = r["tau"]
def agg(c, V):
    ic, cl, tp, fp, fn = (sum(c[v][i] for v in V) for i in range(5)); p = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
    return 100 * ic / max(cl, 1), 2 * p * rr / max(p + rr, 1e-9)
for k in ("qicd", "mc"):
    E = B[(k, "ECT (thresholded, seed 0)")]; V = sorted(E)
    for name in ("No paired neg.", "+ notdone", "+ clean"):
        C = B[(k, name)]; ea, eb = agg(E, V), agg(C, V); rng = random.Random(0); di, df = [], []
        for _ in range(2000):
            S = [rng.choice(V) for _ in V]; x, y = agg(E, S), agg(C, S); di.append(y[0] - x[0]); df.append(y[1] - x[1])
        di.sort(); df.sort()
        got[f"BUDGET/{k}/{name}"] = dict(tau=TAU[(k, name)], ic=round(eb[0], 1), f1=round(eb[1], 3), dIC=round(eb[0] - ea[0], 1), dIC95=[round(di[50], 1), round(di[1949], 1)],
                                      dF1=round(eb[1] - ea[1], 3), dF195=[round(df[50], 3), round(df[1949], 3)],
                                      eqIC=-5 < di[100] and di[1899] < 5, eqF1=-.05 < df[100] and df[1899] < .05)
# ---- long steps
Lg = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in csv.DictReader(open(os.path.join(D, "long_steps_per_video.csv"))):
    Lg[(r["benchmark"], r["group"])][r["run"]][r["video"]] = {k: int(r[k]) for k in ("long_steps", "on_time", "early", "late", "never", "alarm", "other")}
def rate(bench, g, vids, c):
    runs = Lg[(bench, g)]; out = []
    for run in runs.values():
        n = sum(run[v]["long_steps"] for v in vids); out.append(100 * sum(run[v][c] for v in vids) / max(n, 1))
    return sum(out) / len(out)
for bench in ("mc", "qicd"):
    vids = sorted(next(iter(Lg[(bench, "ECT")].values())))
    for c in ("on_time", "early", "alarm"):
        lv = {g: round(rate(bench, g, vids, c), 1) for g in ("ECT", "+ clean", "+ notdone", "no neg.")}
        rng = random.Random(0); Dd = []
        for _ in range(2000):
            sel = [rng.choice(vids) for _ in vids]; Dd.append(rate(bench, "ECT", sel, c) - rate(bench, "+ clean", sel, c))
        Dd.sort(); got[f"LONG/{bench}/{c}"] = dict(levels=lv, diff=round(rate(bench, "ECT", vids, c) - rate(bench, "+ clean", vids, c), 1), ci=[round(Dd[50], 1), round(Dd[1949], 1)])
bad = [k for k in EXP if EXP[k] != got.get(k)]
for k in sorted(got): print(k, got[k])
print("ALL MATCH" if not bad and set(EXP) == set(got) else f"DIFFERENT: {bad}"); sys.exit(1 if bad or set(EXP) != set(got) else 0)
