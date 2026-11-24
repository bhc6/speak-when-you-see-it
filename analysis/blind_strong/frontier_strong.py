"""Margins against the frontier extended by the stronger learned blind policy (blind_strong.py).
Old frontier = analysis_code/frontier_canon.py (hull of single rules, realized hull-edge mixtures and rule mixtures; vertices
measured, interpolation between vertices). New frontier = the same upper hull over the old points plus every single policy of
blind_strong.py, (a) tuned on the scored test set (as the rules are) and (b) validation-selected: only the policies on the
validation hull, scored on test (QICD only; Ego-MC has no validation split, the QICD-validation policies are transferred).
Every row of audit_1114/RESULTS_AUTHORITATIVE.md with a per-step margin is recomputed; the old margins must reproduce.
python frontier_strong.py OUT_DIR"""
import sys, os, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import frontier_canon as F  # noqa: E402
OUT = os.path.abspath(sys.argv[1]); S = json.load(open(os.path.join(OUT, "blind_strong.json")))["splits"]

def upper_hull(pts):
    pts = sorted(set(pts)); h = []
    for x in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (x[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) >= 0: h.pop()
        h.append(x)
    return h
def fr(h, ic):  # identical to frontier_canon.fr, for an arbitrary hull
    best = 0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in h if i >= ic])

# (b) validation-selected policies: the policies on the QICD-validation hull, re-scored on each test set
sys.argv = [sys.argv[0], OUT]; src = open(os.path.join(HERE, "blind_strong.py")).read().split("DS = {")[0]; exec(src)
val = S["neurips2025_validation"]["points"]; vh = set(upper_hull([(p["ic"], p["f1"]) for p in val]))
vsel = [tuple(p["pol"]) for p in val if (p["ic"], p["f1"]) in vh]
HULL = {}
for ds, key in (("qicd", "neurips2025_test"), ("mc", "ego_mc_bench_test")):
    old = list(F.H[ds]); new_t = [(p["ic"], p["f1"]) for p in S[key]["points"]]
    V = SPL[key]; PR = predict(V); new_v = [(r["ic"], r["f1"]) for r in (score(V, PR, pol) for pol in vsel)]
    HULL[ds] = {"old": old, "test_tuned": upper_hull(old + new_t), "val_selected": upper_hull(old + new_v),
             "val_selected_points": sorted(new_v)}

rows = []
for line in open("DATA_ROOT/audit_1114/RESULTS_AUTHORITATIVE.md"):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) >= 8 and c[0] in ("qicd", "mc") and re.fullmatch(r"[+-]?\d+\.\d+", c[7] or ""):
        ic, f1, m = float(c[2]), float(c[5]), float(c[7]); ds = c[0]
        o = round(f1 - F.fr(ds, ic), 3); assert abs(o - m) < 0.0015, (c, o)   # old margins reproduce
        rows.append(dict(set=ds, run=c[1], ic=ic, f1=f1, margin_old=o,
                         margin_test_tuned=round(f1 - fr(HULL[ds]["test_tuned"], ic), 3),
                         margin_val_selected=round(f1 - fr(HULL[ds]["val_selected"], ic), 3)))
json.dump({"hulls": HULL, "rows": rows}, open(os.path.join(OUT, "frontier_strong.json"), "w"), indent=1)
for ds in HULL:
    print(ds, "old", [(round(a, 1), round(b, 3)) for a, b in HULL[ds]["old"]])
    print(ds, "new(test-tuned)", [(round(a, 1), round(b, 3)) for a, b in HULL[ds]["test_tuned"]])
    print(ds, "new(val-selected)", [(round(a, 1), round(b, 3)) for a, b in HULL[ds]["val_selected"]])
for r in rows: print(f"{r['set']:5s} {r['run'][-48:]:48s} {r['ic']:5.1f} {r['f1']:.3f}  old {r['margin_old']:+.3f}  test-tuned {r['margin_test_tuned']:+.3f}  val-sel {r['margin_val_selected']:+.3f}")
