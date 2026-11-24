"""Pre-registered 11-16 23:35: mistake evidence pairs stratified by remaining time R = step_end - t of the positive (mistake)
moment. Strata: end (R < 15 s) / mid (R >= 15 s; same condition as negatives) / mid30 (R >= 30 s). Pair accuracy (ties 0.5),
video bootstrap 2000 for group differences. python epair_stepend.py STEPS.json OUT.json"""
import json, glob, sys, random, collections
st = json.load(open(sys.argv[1]))
T = {}
for m in st["mistakes"]: T[(m["video_id"], m["step"], round(m["t"] - m["start"], 3))] = (m["start"], m["end"], m["t"])
F = {k: f for f in glob.glob("*/results/jobs/epairs/*.json") for k in [f.split("/")[-1][:-5]]}
G = {"ECT": ["T1v4_s0", "T1v4_s1", "T1v4_s2", "T1v4_s3"], "none": ["ABL_none", "ABL_none_s1"], "T1v2": ["T1v2"],
     "clean_only": ["ABL_noND", "ABL_noND_s1"], "notdone_only": ["ABL_noCL", "ABL_noCL_s1", "ABL_noCL_s2", "ABL_noCL_s3"], "R0_zs": ["R0_zs"]}
def strat(R): return ["all"] + (["end"] if R < 15 else ["mid"]) + (["mid30"] if R >= 30 else [])
acc = collections.defaultdict(lambda: collections.defaultdict(list))   # acc[group][(stratum, video)] -> per-pair scores (seed-averaged later)
meta = {}; miss = 0
for g, ks in G.items():
    per = collections.defaultdict(list)
    for k in ks:
        for p in json.load(open(F[k])):
            if p["q"] != "mistake_inference": continue
            key = (p["video_id"], p["pos_step"], round(p["E"], 3))
            if key not in T: miss += 1; continue
            s, e, t = T[key]; R = e - t; meta[key] = R
            per[key].append(1.0 if p["p_pos"] > p["p_neg"] else 0.5 if p["p_pos"] == p["p_neg"] else 0.0)
    for key, v in per.items():
        for sname in strat(meta[key]): acc[g][(sname, key[0])].append(sum(v) / len(v))
vids = sorted({k[0] for k in meta})
def mean(g, s, vs):
    x = [a for v in vs for a in acc[g].get((s, v), [])]; return sum(x) / len(x) if x else float("nan")
rng = random.Random(0); res = {"missing_keys": miss, "n_pairs": {}, "acc": {}, "diff_vs_none": {}}
for s in ["all", "end", "mid", "mid30"]:
    res["n_pairs"][s] = sum(len(acc["ECT"].get((s, v), [])) for v in vids)
    res["acc"][s] = {g: round(mean(g, s, vids), 3) for g in G}
    boots = [[rng.choice(vids) for _ in vids] for _ in range(2000)]
    for g in G:
        if g == "none": continue
        d0 = mean(g, s, vids) - mean("none", s, vids); bs = sorted(mean(g, s, b) - mean("none", s, b) for b in boots)
        bs = [b for b in bs if b == b]
        res["diff_vs_none"].setdefault(s, {})[g] = [round(d0, 3), round(bs[int(.025 * len(bs))], 3), round(bs[int(.975 * len(bs)) - 1], 3)]
    bsE = sorted(mean("ECT", s, b) for b in boots); bsE = [b for b in bsE if b == b]
    res.setdefault("ECT_ci", {})[s] = [round(bsE[int(.025 * len(bsE))], 3), round(bsE[int(.975 * len(bsE)) - 1], 3)]
Rs = sorted(meta.values()); res["R_quantiles"] = [round(Rs[int(q * (len(Rs) - 1))], 1) for q in (0, .25, .5, .75, 1)]
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
