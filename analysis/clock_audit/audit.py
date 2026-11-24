"""Do models before ECT follow the clock? CPU-only audit of frozen test predictions (runs.py; each re-scored first).
Per reached mistake-free step (segment = outputs after its Instruction line; elapsed e = time since that line):
  A  first false alarm: share at the first check (e <= 5.5 s) and the hazard per check (P(first alarm at check k | none
     before and not yet confirmed)), checks binned 1 / 2-3 / 4-6 / 7-12 / 13+.
  B  first confirmation vs the step's annotated duration D and the training prior Dp (median annotated duration of the
     same instruction in the QICD train split, global median if unseen). log-scale Pearson r(c, D), r(c, Dp), and the
     partial correlations r(c, D | Dp) and r(c, Dp | D); 95% intervals by bootstrap over videos (1000). A clock-following
     policy confirms at the prior: r(c, Dp | D) > 0 and r(c, D | Dp) ~ 0. Ego-MC instructions are almost never seen in
     QICD train, so there the prior is the global median and only r(c, D) is informative.
  C  steps with a mistake: first alarm relative to the first annotated mistake (> 15 s before / within 15 s / > 15 s after).
python3 audit.py -> out/audit.json"""
import sys, os, json, types, random, math, statistics as st, collections
_t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
V = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V + "analysis_code/blind_strong"); sys.path.insert(0, HERE)
import local_eval as L
from runs import RUNS
OUT = HERE + "/out"; os.makedirs(OUT, exist_ok=True)
GT = json.load(open(V + "analysis_code/blind_strong/inputs/gt_dump_1118.json"))
norm = lambda s: s.replace("Instruction:", "").strip()
prior = collections.defaultdict(list)
for g in GT["neurips2025_train"]:
    for tx, ts in zip(g["instruction_segment_texts"], g["instruction_segment_texts_timestamps"]):
        if max(ts) > ts[0]: prior[norm(tx[0])].append(max(ts) - ts[0])
PRIOR = {k: st.median(v) for k, v in prior.items()}; GMED = st.median([x for v in prior.values() for x in v])
BINS = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 999)]


def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruction"): out.append([])
        if out: out[-1].append((t, ts))
    return out


def pearson(x, y):
    if len(x) < 4: return float("nan")
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def partial(x, y, z):
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    d = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12)); return (rxy - rxz * ryz) / d


def corr_stats(rows):  # rows: (video, log c, log D, log Dp)
    def f(R):
        c, D, P = [r[1] for r in R], [r[2] for r in R], [r[3] for r in R]
        return dict(r_D=pearson(c, D), r_P=pearson(c, P), r_D_given_P=partial(c, D, P), r_P_given_D=partial(c, P, D))
    est = f(rows); vids = sorted({r[0] for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r[0]].append(r)
    rng = random.Random(0); boots = collections.defaultdict(list)
    for _ in range(1000):
        R = [r for v in (rng.choice(vids) for _ in vids) for r in by[v]]
        for k, x in f(R).items():
            if not math.isnan(x): boots[k].append(x)
    ci = {k: [round(sorted(v)[int(.025 * len(v))], 3), round(sorted(v)[int(.975 * len(v)) - 1], 3)] for k, v in boots.items()}
    return {k: [round(v, 3), ci.get(k)] for k, v in est.items()}


res = {}
for b, runs in RUNS.items():
    dk = "neurips2025_test" if b == "qicd" else "ego_mc_bench_test"
    ds = L.load(dk.rsplit("_", 1)[0], "test"); G = {g["video_id"]: g for g in GT[dk]}
    for name, grp, path, ic, f1 in runs:
        P = json.load(open(path)); s, _ = L.run(ds, P)
        assert abs(s["ic"] - ic) < 0.06 and abs(s["f1"] - f1) < 0.0015, (b, name, s)
        first_fa, at_risk, hit, crows, mis = [], collections.Counter(), collections.Counter(), [], collections.Counter()
        seen = 0
        for p in P:
            g = G[p["video_id"]]; S = segs(p)
            for i, (tx, ts, ty) in enumerate(zip(g["instruction_segment_texts"], g["instruction_segment_texts_timestamps"], g["instruction_segment_texts_types"])):
                if i >= len(S): continue
                st0, D = S[i][0][1], max(ts) - ts[0]
                fb = [x - st0 for t, x in S[i] if t.startswith("Feedback")]; sc = [x - st0 for t, x in S[i] if t.startswith("Success")]
                if g["instrcution_segment_has_mistake"][i]:
                    m = [x for x, t in zip(ts, ty) if "mistake" in t]
                    if fb and m:
                        d = (fb[0] + st0) - min(m); mis["before" if d < -15 else "within" if d <= 15 else "after"] += 1
                    continue
                seen += 1
                if fb: first_fa.append(fb[0])
                # hazard of the first alarm per check, while the step is open (no alarm yet, not confirmed)
                stop = min(fb[:1] + sc[:1] + [1e9]); last = max([x for _, x in S[i]] + [0]) - st0
                for k in range(1, int(min(stop, last) / 5 + 0.5) + 1):
                    bk = next(j for j, (lo, hi) in enumerate(BINS) if lo <= k <= hi); at_risk[bk] += 1
                    if fb and abs(fb[0] - 5 * k) < 2.6 and fb[0] <= (sc[0] if sc else 1e9): hit[bk] += 1
                if sc and D > 0 and sc[0] > 0:
                    dp = PRIOR.get(norm(tx[0]), GMED); crows.append((p["video_id"], math.log(sc[0]), math.log(D), math.log(dp)))
        res[f"{b}|{name}"] = dict(group=grp, ic=s["ic"], f1=s["f1"], clean_reached=seen, false_alarm_steps=len(first_fa),
            fa_first_check=round(sum(x <= 5.5 for x in first_fa) / max(len(first_fa), 1), 3),
            fa_median_s=round(st.median(first_fa), 1) if first_fa else None,
            hazard={f"{lo}-{hi}": round(hit[j] / at_risk[j], 3) if at_risk[j] else None for j, (lo, hi) in enumerate(BINS)},
            confirmed=len(crows), confirm_corr=corr_stats(crows) if len(crows) > 10 else None,
            mistake_alarm_vs_first_mistake=dict(mis))
        r = res[f"{b}|{name}"]; cc = r["confirm_corr"] or {}
        print(f"{b:4s} {name:30s} FA steps {r['false_alarm_steps']:3d}/{seen:3d} first-check {r['fa_first_check']:.0%} med {r['fa_median_s']} "
              f"haz {list(r['hazard'].values())} | conf {len(crows)} r(D) {cc.get('r_D')} r(P) {cc.get('r_P')} r(D|P) {cc.get('r_D_given_P')} "
              f"r(P|D) {cc.get('r_P_given_D')} | mis {dict(mis)}", flush=True)
prior_cov = {b: round(sum(norm(tx[0]) in PRIOR for g in GT[k] for tx in g["instruction_segment_texts"]) /
                      sum(len(g["instruction_segment_texts"]) for g in GT[k]), 3) for b, k in (("qicd", "neurips2025_test"), ("mc", "ego_mc_bench_test"))}
print("share of test steps whose instruction has a train prior:", prior_cov)
json.dump(dict(prior_coverage=prior_cov, runs=res), open(OUT + "/audit.json", "w"), indent=1)
