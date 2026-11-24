"""Pre-registered 11-16 23:45: step-end-matched mistake control on the validation situation probe (*.auroc.json).
mistake moments vs completion moments of mistake-free steps (both at step end); P(yes) of the mistake question.
Main metric: AUROC within 10 s elapsed-time bins, weighted by #pos*#neg. python stepend_control.py VAL_STEPS.json OUT.json"""
import json, sys, random, collections
st = json.load(open(sys.argv[1]))["steps"]
M = {"ECT_s0": "results/auroc_r5/T1v4_s2400.json", "ECT_s1": "machine4/results/jobs/T1v4_seed1.auroc.json", "ECT_s2": "machine5/results/jobs/T1v4_seed2.auroc.json",
     "none": "machine4/results/jobs/ABL_none.auroc.json", "clean_only": "machine0/results/jobs/ABL_noND.auroc.json", "notdone_only": "machine7/results/jobs/ABL_noCL.auroc.json"}
D = {k: json.load(open(f)) for k, f in M.items()}
def auc(P, N):
    if not P or not N: return None
    s = sum((p > n) + 0.5 * (p == n) for p in P for n in N); return s / (len(P) * len(N))
def metrics(d, vids):
    vs = collections.Counter(vids); rows = [x for x in d if x["video_id"] in vs]
    W = []
    for x in rows: W += [x] * vs[x["video_id"]]
    mis = [x for x in W if x["kind"] == "mistake"]; suc = [x for x in W if x["kind"] == "success"]; sil = [x for x in W if x["kind"] == "silence"]
    b = lambda x: int(x["elapsed"] // 10); num = den = 0
    bm, bs = collections.defaultdict(list), collections.defaultdict(list)
    for x in mis: bm[b(x)].append(x["p_mis"])
    for x in suc: bs[b(x)].append(x["p_mis"])
    for k in bm:
        if k in bs: a = auc(bm[k], bs[k]); w = len(bm[k]) * len(bs[k]); num += a * w; den += w
    return dict(end_matched=num / den if den else None, mis_vs_success=auc([x["p_mis"] for x in mis], [x["p_mis"] for x in suc]),
                mis_vs_silence=auc([x["p_mis"] for x in mis], [x["p_mis"] for x in sil]), matched_frac_pos=sum(len(bm[k]) for k in bm if k in bs) / len(mis))
vids = sorted({x["video_id"] for x in D["none"]})
R = sorted(st[x["video_id"]][str(x["step"])][1] - x["t"] for x in D["none"] if x["kind"] == "mistake")
res = {"n": {k: sum(x["kind"] == k for x in D["none"]) for k in ["mistake", "success", "silence"]}, "mistake_R_quartiles": [round(R[int(q * (len(R) - 1))], 1) for q in (0, .25, .5, .75, 1)],
       "point": {k: {m: round(v, 3) for m, v in metrics(d, vids).items()} for k, d in D.items()}}
rng = random.Random(0); boots = [[rng.choice(vids) for _ in vids] for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2000)]
bt = {k: [metrics(d, b) for b in boots] for k, d in D.items()}
ci = lambda xs: [round(sorted(xs)[int(.025 * len(xs))], 3), round(sorted(xs)[int(.975 * len(xs)) - 1], 3)]
res["ci"] = {k: {m: ci([r[m] for r in bt[k]]) for m in ["end_matched", "mis_vs_success", "mis_vs_silence"]} for k in D}
ect = lambda i, m: sum(bt[k][i][m] for k in ["ECT_s0", "ECT_s1", "ECT_s2"]) / 3
res["ECT_mean_minus"] = {g: {m: ci([ect(i, m) - bt[g][i][m] for i in range(len(boots))]) for m in ["end_matched", "mis_vs_silence"]} for g in ["none", "clean_only", "notdone_only"]}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
