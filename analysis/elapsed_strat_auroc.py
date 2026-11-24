"""Elapsed-time-stratified AUROC on the shared validation situations (auroc_probe output, CPU).
completion: success vs in-progress (silence, shifted) under P(done); mistake: mistake vs non-mistake under P(mis).
Pairs only compared within the same elapsed bin (10 s) -> elapsed time cannot help; video bootstrap CI.
Also elapsed-only AUROC (how much a pure timer would get) and plain AUROC."""
import json, sys, random, collections
def auc(pairs):
    s = n = 0
    for p, q in pairs: s += (p > q) + 0.5 * (p == q); n += 1
    return s / max(n, 1)
def strat(rows, pos, neg, key, B=10):
    by = collections.defaultdict(lambda: ([], []))
    for r in rows:
        b = int(r["elapsed"] // B)
        if r["kind"] in pos: by[b][0].append(r)
        elif r["kind"] in neg: by[b][1].append(r)
    pr = [(p[key], q[key]) for P, N in by.values() for p in P for q in N]
    return pr
def run(f):
    R = json.load(open(f)); vids = sorted({r["video_id"] for r in R}); rng = random.Random(0); out = {}
    for name, pos, neg, key in (("completion", {"success"}, {"silence", "shifted"}, "p_done"), ("mistake", {"mistake"}, {"silence", "shifted", "success"}, "p_mis")):
        full = auc(strat(R, pos, neg, key)); plain = auc([(p[key], q[key]) for p in R if p["kind"] in pos for q in R if q["kind"] in neg])
        timer = auc([(p["elapsed"], q["elapsed"]) for p in R if p["kind"] in pos for q in R if q["kind"] in neg])
        bs = []
        byv = collections.defaultdict(list)
        for r in R: byv[r["video_id"]].append(r)
        for _ in range(200):
            S = [r for v in (rng.choice(vids) for _ in vids) for r in byv[v]]; bs.append(auc(strat(S, pos, neg, key)))
        bs.sort(); out[name] = (round(plain, 3), round(full, 3), round(bs[5], 3), round(bs[194], 3), round(timer, 3))
    return out
for f in sys.argv[1:]:
    o = run(f); print(f.split("/")[-1].replace(".auroc.json", "").replace(".json", ""), "| completion plain/stratified [CI] timer:", o["completion"], "| mistake:", o["mistake"])
