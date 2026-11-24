"""Summary of the stronger learned blind policy for the supplement: margins against the paper's frontier (frontier_canon /
frontier_main: rules + realized mixtures, tuned on the test set) and against the upper hull of that frontier together with each
learned family (strict / nohist / full, tuned on the test set); half-split out-of-sample margins from verify_strong*.json
(rules without mixtures + learned family). Also every other row with a margin (published, zero-shot, ablations).
python summarize_strong.py OUT_DIR"""
import sys, os, json, re, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); D = "DATA_ROOT"; sys.path.insert(0, f"{D}/analysis_code")
import frontier_canon as F
OUT = os.path.abspath(sys.argv[1])
sys.argv = ["x", f"{D}/machine4/results"]
with contextlib.redirect_stdout(io.StringIO()): exec(open(f"{D}/analysis_code/frontier_main.py").read().split("res = dict(")[0])
MAINH = h; frmain = fr; D = "DATA_ROOT"
def hull(pts):
    pts = sorted(set(pts)); H = []
    for x in pts:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (x[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (x[0] - H[-2][0]) >= 0: H.pop()
        H.append(x)
    return H
def frh(H, ic):
    best = 0
    for (a, fa), (b, fb) in zip(H, H[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in H if i >= ic])
V = {"qicd": json.load(open(f"{OUT}/verify_strong.json")), "mc": json.load(open(f"{OUT}/verify_strong.json")), "main": json.load(open(f"{OUT}/verify_strong_main.json"))}
CANON = {"qicd": list(F.H["qicd"]), "mc": list(F.H["mc"]), "main": list(MAINH)}
CFR = {"qicd": lambda ic: F.fr("qicd", ic), "mc": lambda ic: F.fr("mc", ic), "main": frmain}
fam_key = {"qicd": "qicd", "mc": "mc", "main": "qicd"}
HUL = {s: {f: hull(CANON[s] + [(float(a), float(b)) for a, b, _ in V[s]["families"][f"{fam_key[s]}/{f}"]["hull"]]) for f in ("strict", "nohist", "full")} for s in V}
# ECT points
PTS = {"qicd": {"greedy": [(41.2, .217), (39.9, .284), (39.5, .267), (42.4, .221)], "thr": [(42.0, .306), (41.2, .322), (41.6, .303), (44.1, .242)]},
       "mc": {"greedy": [(67.0, .208), (54.2, .228), (59.1, .278), (55.7, .271)], "thr": [(55.3, .203), (48.1, .211), (53.0, .274), (53.8, .280)]},
       "main": {"greedy": [(53.5, .202), (47.4, .224), (50.9, .212), (56.0, .192)], "thr": [(41.2, .203), (40.3, .223), (41.7, .229), (46.1, .205)]}}
res = {"hulls": HUL, "ect": {}, "others": {}}
for s in PTS:
    for k, L in PTS[s].items():
        for i, (ic, f1) in enumerate(L):
            cf = V[s]["crossfit"][fam_key[s]]
            r = dict(ic=ic, f1=f1, paper=round(f1 - CFR[s](ic), 3), **{f: round(f1 - frh(HUL[s][f], ic), 3) for f in HUL[s]},
                     oos_strict=cf["strict"]["ect"][f"{k}_s{i}"]["oos_mean"], oos_full=cf["full"]["ect"][f"{k}_s{i}"]["oos_mean"],
                     p_strict=cf["strict"]["ect"][f"{k}_s{i}"]["oos_p_beyond"], p_full=cf["full"]["ect"][f"{k}_s{i}"]["oos_p_beyond"])
            res["ect"][f"{s}/{k}/s{i}"] = r
# every other row with a margin (authoritative table; main: published + our main runs)
for line in open(f"{D}/audit_1114/RESULTS_AUTHORITATIVE.md"):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) >= 8 and c[0] in ("qicd", "mc") and re.fullmatch(r"[+-]?\d+\.\d+", c[7] or ""):
        ic, f1 = float(c[2]), float(c[5]); res["others"][f"{c[0]}/{c[1]}"] = dict(ic=ic, f1=f1, paper=round(f1 - CFR[c[0]](ic), 3), strict=round(f1 - frh(HUL[c[0]]["strict"], ic), 3), full=round(f1 - frh(HUL[c[0]]["full"], ic), 3))
for name, ic, f1 in PUB:
    res["others"][f"main/published/{name}"] = dict(ic=ic, f1=f1, paper=round(f1 - frmain(ic), 3), strict=round(f1 - frh(HUL["main"]["strict"], ic), 3), full=round(f1 - frh(HUL["main"]["full"], ic), 3))
json.dump(res, open(f"{OUT}/summary_strong.json", "w"), indent=1)
for k, r in res["ect"].items(): print(f"{k:16s} {r['ic']:5.1f}/{r['f1']:.3f} paper {r['paper']:+.3f} strict {r['strict']:+.3f} nohist {r['nohist']:+.3f} full {r['full']:+.3f} | out-of-sample strict {r['oos_strict']:+.3f} ({r['p_strict']:.2f}) full {r['oos_full']:+.3f} ({r['p_full']:.2f})")
neg = [(k, r) for k, r in res["others"].items() if r["paper"] <= 0]
print("rows inside the paper frontier:", len(neg), "-> inside strict:", sum(r["strict"] <= 0 for _, r in neg), "inside full:", sum(r["full"] <= 0 for _, r in neg))
pos = [(k, r) for k, r in res["others"].items() if r["paper"] > 0]
print("rows beyond the paper frontier:", len(pos), "-> still beyond strict:", sum(r["strict"] > 0 for _, r in pos), "full:", sum(r["full"] > 0 for _, r in pos))
for k, r in sorted(res["others"].items()):
    if "published" in k or "API_q38max_mc" in k or "397a17_mc" in k: print(f"  {k:55s} paper {r['paper']:+.3f} strict {r['strict']:+.3f} full {r['full']:+.3f}")
