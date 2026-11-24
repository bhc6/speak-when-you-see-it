"""Pre-registered 11-16 19:55: step-aligned swap (ECT s0, QICD test36). Matched steps where BOTH the original and the donor
step are mistake-free. Confirmation time = elapsed (s) since own step start at the first 'yes' to the completion question.
r(confirm, donor duration), r(confirm, own duration); joint OLS confirm ~ 1 + donor + own (video bootstrap 2000);
IC-style hit (single confirmation within +-15 s of the step end, no Feedback before it) against own vs donor durations.
python swapalign_analysis.py STEPS.json OUT.json"""
import json, glob, sys, random, collections
def corr(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b)); return sab / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
def ols2(y, x1, x2):  # y ~ 1 + x1 + x2 via centred normal equations
    n = len(y); m = [sum(v) / n for v in (y, x1, x2)]; c = lambda a, ma, b, mb: sum((p - ma) * (q - mb) for p, q in zip(a, b))
    s11, s22, s12 = c(x1, m[1], x1, m[1]), c(x2, m[2], x2, m[2]), c(x1, m[1], x2, m[2]); s1y, s2y = c(x1, m[1], y, m[0]), c(x2, m[2], y, m[0])
    det = s11 * s22 - s12 ** 2; return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det
def pct(xs, q): xs = sorted(xs); return xs[min(int(q / 100 * len(xs)), len(xs) - 1)]
st = json.load(open(sys.argv[1])); S = st["steps"]
mis = {(m["video_id"], m["step"]) for m in st["mistakes"]}
R = [json.loads(l) for f in sorted(glob.glob("box*/results/jobs/SWAPALIGN_ECT_s0_test36/align*.jsonl")) for l in open(f)]
yes = lambda o: o.strip().lower().startswith("yes")
steps = collections.OrderedDict()
for r in R:
    if not r["matched"]: continue
    k = (r["video"], r["own_step"]); d = steps.setdefault(k, dict(donor=r["donor"], dstep=r["donor_step"], conf=None, fb=False, n=0))
    if d["conf"] is not None: continue
    s0 = S[r["video"]][str(r["own_step"])][0]; e = r["end"] / 2 - s0
    if r["mode"] == "instruction_end":
        d["n"] += 1
        if yes(r["out"]): d["conf"] = e
    elif r["mode"] == "mistake_inference" and yes(r["out"]): d["fb"] = True
rows = []; excl = collections.Counter()
for (v, i), d in steps.items():
    if str(i) not in S[v] or str(d["dstep"]) not in S[d["donor"]]: excl["no_boundary"] += 1; continue
    if (v, i) in mis or (d["donor"], d["dstep"]) in mis: excl["mistake_step"] += 1; continue
    os_, oe = S[v][str(i)]; ds, de = S[d["donor"]][str(d["dstep"])]
    rows.append(dict(video=v, step=i, own=oe - os_, donor=de - ds, conf=d["conf"], fb=d["fb"]))
res = dict(n_matched_steps=len(steps), excluded=dict(excl), n_both_clean=len(rows), n_videos=len({r["video"] for r in rows}))
C = [r for r in rows if r["conf"] is not None]; res["n_confirmed"] = len(C)
res["dur_corr_own_donor"] = round(corr([r["own"] for r in rows], [r["donor"] for r in rows]), 3)
def stats(rs):
    c = [r for r in rs if r["conf"] is not None]
    y = [r["conf"] for r in c]; dd = [r["donor"] for r in c]; oo = [r["own"] for r in c]; b = (None,) + ols2(y, dd, oo)
    hit = lambda key: sum(1 for r in rs if r["conf"] is not None and not r["fb"] and abs(r["conf"] - r[key]) <= 15) / len(rs)
    return dict(r_donor=corr(y, dd), r_own=corr(y, oo), b_donor=float(b[1]), b_own=float(b[2]),
                ic_own=100 * hit("own"), ic_donor=100 * hit("donor"))
pt = stats(rows); res["point"] = {k: round(v, 3) for k, v in pt.items()}
vids = sorted({r["video"] for r in rows}); by = collections.defaultdict(list)
for r in rows: by[r["video"]].append(r)
rng = random.Random(0); B = []
for _ in range(2000):
    rs = [r for v in (rng.choice(vids) for _ in vids) for r in by[v]]
    try: s = stats(rs); s["d_r"] = s["r_donor"] - s["r_own"]; s["d_b"] = s["b_donor"] - s["b_own"]; s["d_ic"] = s["ic_donor"] - s["ic_own"]; B.append(s)
    except Exception: pass
res["ci95"] = {k: [round(pct([b[k] for b in B], 2.5), 3), round(pct([b[k] for b in B], 97.5), 3)] for k in B[0]}
res["n_boot"] = len(B)
json.dump(dict(res, rows=rows), open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
