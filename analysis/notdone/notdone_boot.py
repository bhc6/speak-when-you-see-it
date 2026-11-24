"""Robustness of the notdone findings on Ego-MC-Bench (post-hoc): per-seed values and a paired bootstrap over videos (2000) for
the difference "mean of ECT seeds minus mean of '+ clean' seeds" in (a) the partial correlation of confirmation time with the
annotated duration given the training prior (global median: Ego-MC instructions are unseen), (b) the on-time rate on the longest
third of the mistake-free steps, (c) the early-confirmation rate on that third; same for "+ notdone" and "no neg." against "+ clean".
Uses notdone_stats.py (same step sets and classes). python notdone_boot.py OUT_DIR"""
import sys, os, json, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = sys.argv[1]
src = open(os.path.join(HERE, "notdone_stats.py")).read().split("res = {}")[0]; sys.argv = ["x", OUTD]; exec(src)
OUT = os.path.abspath(OUTD)
ds = L.load("ego_mc_bench", "test"); groups = RUNS["mc"]
S = {g: [steps_of(ds, json.load(open(B + p + "/pred.json"))) for p in ps] for g, ps in groups.items()}
keys = sorted(set.intersection(*[set(s) for runs in S.values() for s in runs]))
q = sorted(S["ECT"][0][k]["dur"] for k in keys); cut = q[2 * len(q) // 3]; long_ = {k for k in keys if S["ECT"][0][k]["dur"] >= cut}
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(sa * sb, 1e-9)
def metrics(run, K):
    xs = [(run[k]["offset"] + run[k]["dur"], run[k]["dur"]) for k in K if not run[k]["alarm"] and run[k]["offset"] is not None]
    r = corr([x[0] for x in xs], [x[1] for x in xs]) if len(xs) > 2 else float("nan")   # prior is constant on Ego-MC: partial = plain
    L_ = [k for k in K if k in long_]; c = [cls(run[k]) for k in L_]
    return r, 100 * c.count("on time") / max(len(c), 1), 100 * c.count("early") / max(len(c), 1)
per_seed = {g: [tuple(round(x, 3) for x in metrics(r, keys)) for r in runs] for g, runs in S.items()}
for g, v in per_seed.items(): print(f"{g:10s} per seed (corr, on-time long %, early long %):", v)
vids = sorted({k[0] for k in keys}); byv = {v: [k for k in keys if k[0] == v] for v in vids}
def gmean(g, K): M = [metrics(r, K) for r in S[g]]; return [st.mean(m[i] for m in M) for i in range(3)]
res = {"per_seed": per_seed, "n_steps": len(keys), "n_long": len(long_), "long_cut_s": cut, "diffs": {}}
for a, b in (("ECT", "+ clean"), ("+ notdone", "+ clean"), ("+ notdone", "no neg."), ("ECT", "no neg.")):
    full = [x - y for x, y in zip(gmean(a, keys), gmean(b, keys))]; rng = random.Random(0); D = [[], [], []]
    for _ in range(2000):
        K = [k for v in (rng.choice(vids) for _ in vids) for k in byv[v]]
        for i, d in enumerate(x - y for x, y in zip(gmean(a, K), gmean(b, K))): D[i].append(d)
    ci = [(sorted(d)[50], sorted(d)[1949]) for d in D]
    res["diffs"][f"{a} - {b}"] = dict(corr=(round(full[0], 3), [round(x, 3) for x in ci[0]]), ontime_long=(round(full[1], 1), [round(x, 1) for x in ci[1]]),
                                      early_long=(round(full[2], 1), [round(x, 1) for x in ci[2]]))
    print(f"{a} - {b}: corr {full[0]:+.3f} [{ci[0][0]:+.3f}, {ci[0][1]:+.3f}] | on time (long) {full[1]:+.1f} [{ci[1][0]:+.1f}, {ci[1][1]:+.1f}] | early (long) {full[2]:+.1f} [{ci[2][0]:+.1f}, {ci[2][1]:+.1f}]", flush=True)
json.dump(res, open(f"{OUT}/notdone_boot.json", "w"), indent=1)
