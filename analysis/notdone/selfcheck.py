"""Self-check of the notdone analysis (post-hoc). (1) correlation on the common set of steps that every run confirms without an
alarm; (2) time expressions: strict (explicit time units) vs broad (also 'until/till', a completion criterion); (3) the same absolute
long-step definition (top third of annotated duration) on QICD and Ego-MC; bootstrap over videos (2000), ECT mean minus '+ clean' mean.
python selfcheck.py OUT_DIR"""
import sys, os, json, re, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = sys.argv[1]
src = open(os.path.join(HERE, "notdone_stats.py")).read().split("res = {}")[0]; sys.argv = ["x", OUTD]; exec(src); OUT = os.path.abspath(OUTD)
STRICT = re.compile(r"\b(minute|minutes|min|second|seconds|sec|hour|hours)\b", re.I); BROAD = re.compile(r"\b(minute|minutes|min|second|seconds|sec|hour|until|till|for about|for around)\b", re.I)
def corr(a, b):
    if len(a) < 3: return float("nan")
    ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(sa * sb, 1e-9)
def boot(f, vids, K, n=2000):
    rng = random.Random(0); D = []
    for _ in range(n):
        sel = [rng.choice(vids) for _ in vids]; KK = [k for v in sel for k in K if k[0] == v]; d = f(KK)
        if d == d: D.append(d)
    D.sort(); return [round(D[int(.025 * len(D))], 3), round(D[int(.975 * len(D)) - 1], 3)]
res = {}
for dsn in ("mc", "qicd"):
    ds = L.load("ego_mc_bench" if dsn == "mc" else "neurips2025", "test"); S = {g: [steps_of(ds, json.load(open(B + p + "/pred.json"))) for p in ps] for g, ps in RUNS[dsn].items()}
    keys = sorted(set.intersection(*[set(s) for runs in S.values() for s in runs])); vids = sorted({k[0] for k in keys}); E0 = S["ECT"][0]
    R = {}
    # (1) common confirmed set
    common = [k for k in keys if all(not r[k]["alarm"] and r[k]["offset"] is not None for runs in S.values() for r in runs)]
    cc = lambda g, K: st.mean(corr([r[k]["offset"] + r[k]["dur"] for k in K], [r[k]["dur"] for k in K]) for r in S[g])
    R["common_confirmed"] = dict(n=len(common), **{g: round(cc(g, common), 3) for g in S}, ect_minus_clean=round(cc("ECT", common) - cc("+ clean", common), 3),
                                 ci=boot(lambda K: cc("ECT", K) - cc("+ clean", K), vids, common))
    # (2) time expressions
    R["n_time_broad"] = sum(bool(BROAD.search(E0[k]["instr"])) for k in keys); R["n_time_strict"] = sum(bool(STRICT.search(E0[k]["instr"])) for k in keys)
    R["until_only"] = [E0[k]["instr"][:90] for k in keys if BROAD.search(E0[k]["instr"]) and not STRICT.search(E0[k]["instr"])]
    q = sorted(E0[k]["dur"] for k in keys); cut = q[2 * len(q) // 3]; LONG = [k for k in keys if E0[k]["dur"] >= cut]
    def rate(g, K, c): return st.mean(100 * sum(cls(r[k]) == c for k in K) / max(len(K), 1) for r in S[g])
    for name, rx in (("strict", STRICT), ("broad", BROAD)):
        NTL = [k for k in LONG if not rx.search(E0[k]["instr"])]; TE = [k for k in keys if rx.search(E0[k]["instr"])]
        R[f"long_no_time_{name}"] = dict(n=len(NTL), ect=round(rate("ECT", NTL, "on time"), 1), clean=round(rate("+ clean", NTL, "on time"), 1),
                                         diff=round(rate("ECT", NTL, "on time") - rate("+ clean", NTL, "on time"), 1), ci=boot(lambda K: rate("ECT", [k for k in K if k in set(NTL)], "on time") - rate("+ clean", [k for k in K if k in set(NTL)], "on time"), vids, NTL))
        R[f"time_expr_early_{name}"] = dict(n=len(TE), ect=round(rate("ECT", TE, "early"), 1), clean=round(rate("+ clean", TE, "early"), 1),
                                            diff=round(rate("ECT", TE, "early") - rate("+ clean", TE, "early"), 1), ci=boot(lambda K: rate("ECT", K, "early") - rate("+ clean", K, "early"), vids, TE))
    # (3) same absolute long-step definition
    R["long_abs"] = dict(cut_s=round(cut, 1), n=len(LONG), per_seed={g: [round(100 * sum(cls(r[k]) == "on time" for k in LONG) / len(LONG), 1) for r in S[g]] for g in S},
                         diff=round(rate("ECT", LONG, "on time") - rate("+ clean", LONG, "on time"), 1), ci=boot(lambda K: rate("ECT", K, "on time") - rate("+ clean", K, "on time"), vids, LONG))
    res[dsn] = R
    print(f"== {dsn}"); [print("  ", k, v) for k, v in R.items() if k != "until_only"]; print("   'until'-only instructions:", len(R["until_only"]), R["until_only"][:6])
json.dump(res, open(f"{OUT}/selfcheck.json", "w"), indent=1)
