"""Is the notdone effect on Ego-MC-Bench read from the instruction text? Steps whose instruction states a time ("cook for about two
minutes", "until ...") are separated from the others; paired bootstrap over videos (2000), mean of ECT seeds minus mean of
"+ clean" seeds (and "+ notdone" minus "+ clean"): correlation of confirmation time with the annotated duration on steps without
a time expression; on-time and early rates on the longest third of the steps without a time expression; on-time and early rates
on the steps with a time expression. Post-hoc, exploratory. python notdone_textcue.py OUT_DIR"""
import sys, os, json, re, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = sys.argv[1]
src = open(os.path.join(HERE, "notdone_stats.py")).read().split("res = {}")[0]; sys.argv = ["x", OUTD]; exec(src); OUT = os.path.abspath(OUTD)
TIME = re.compile(r"\b(minute|minutes|min|second|seconds|sec|hour|until|till|for about|for around)\b", re.I)
ds = L.load("ego_mc_bench", "test"); S = {g: [steps_of(ds, json.load(open(B + p + "/pred.json"))) for p in ps] for g, ps in RUNS["mc"].items()}
keys = sorted(set.intersection(*[set(s) for runs in S.values() for s in runs])); T = {k: bool(TIME.search(S["ECT"][0][k]["instr"])) for k in keys}
q = sorted(S["ECT"][0][k]["dur"] for k in keys); cut = q[2 * len(q) // 3]
NT = [k for k in keys if not T[k]]; NTL = [k for k in NT if S["ECT"][0][k]["dur"] >= cut]; TE = [k for k in keys if T[k]]
def corr(a, b):
    if len(a) < 3: return float("nan")
    ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(sa * sb, 1e-9)
def m(g, K, KL, KT):
    out = []
    for r in S[g]:
        xs = [(r[k]["offset"] + r[k]["dur"], r[k]["dur"]) for k in K if not r[k]["alarm"] and r[k]["offset"] is not None]
        cl_ = [cls(r[k]) for k in KL]; ct = [cls(r[k]) for k in KT]
        out.append((corr([x[0] for x in xs], [x[1] for x in xs]), 100 * cl_.count("on time") / max(len(cl_), 1), 100 * cl_.count("early") / max(len(cl_), 1),
                    100 * ct.count("on time") / max(len(ct), 1), 100 * ct.count("early") / max(len(ct), 1)))
    return [st.mean(x[i] for x in out) for i in range(5)]
names = ["corr, no time expr", "on time, long, no time expr", "early, long, no time expr", "on time, time expr", "early, time expr"]
vids = sorted({k[0] for k in keys}); by = lambda K, sel: [k for v in sel for k in K if k[0] == v]
res = dict(n=len(keys), n_no_time=len(NT), n_long_no_time=len(NTL), n_time=len(TE), levels={g: [round(x, 3) for x in m(g, NT, NTL, TE)] for g in S}, diffs={})
for a, b in (("ECT", "+ clean"), ("+ notdone", "+ clean")):
    full = [x - y for x, y in zip(m(a, NT, NTL, TE), m(b, NT, NTL, TE))]; rng = random.Random(0); D = [[] for _ in names]
    for _ in range(2000):
        sel = [rng.choice(vids) for _ in vids]; d = [x - y for x, y in zip(m(a, by(NT, sel), by(NTL, sel), by(TE, sel)), m(b, by(NT, sel), by(NTL, sel), by(TE, sel)))]
        for i, x in enumerate(d):
            if x == x: D[i].append(x)
    res["diffs"][f"{a} - {b}"] = {n: (round(f, 3), [round(sorted(d)[int(.025 * len(d))], 3), round(sorted(d)[int(.975 * len(d)) - 1], 3)]) for n, f, d in zip(names, full, D)}
print(json.dumps({k: v for k, v in res.items() if k != "levels"}, indent=1)); print(json.dumps(res["levels"]))
json.dump(res, open(f"{OUT}/notdone_textcue.json", "w"), indent=1)
