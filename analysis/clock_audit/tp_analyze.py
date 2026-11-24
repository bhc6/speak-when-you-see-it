"""Summary of gpu/time_probe.py outputs (machine5/results/clk1122/tp_*.json), on the 411 situations shared by all four runs
(--stride 9 of the probe's sorted situation list; ECT s0 holds 33 extra stride-3 situations from an earlier start, dropped).
Per model and time-stamp condition (ts2 = every stamp doubled, ts05 = halved; pixels, frame count and prompt unchanged):
mean change of P(yes) against the official stamps (signed and absolute), and the completion AUROC (success vs silence) and
mistake AUROC (mistake vs silence), with 95% intervals from 1000 bootstrap resamples of validation videos.
python3 tp_analyze.py -> out/tp_analyze.json"""
import json, os, random, bisect, statistics as st, collections
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
SRC = V0 + "machine5/results/clk1122/"; OUT = HERE + "/out"; os.makedirs(OUT, exist_ok=True)
RUNS = [("ECT s0", "tp_ECT_s0"), ("no paired neg. s0", "tp_noneg_s0"), ("plain fine-tuning", "tp_plain"), ("ECT recipe on black frames s0", "tp_blind_s0")]
key = lambda r: (r["video_id"], r["step"], r["kind"], r["t"])
D = {n: json.load(open(SRC + f + ".json")) for n, f in RUNS}
K = set.intersection(*[{key(r) for r in d} for d in D.values()])


def auc(p, n):
    n = sorted(n); return sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p) / max(len(p) * len(n), 1)


def stats(R):
    o = {}
    for q in ("done", "mis"):
        for c in ("ts2", "ts05"):
            o[f"{q}_{c}_shift"] = st.mean(r[f"{q}_{c}"] - r[f"{q}_ts1"] for r in R)
            o[f"{q}_{c}_abs"] = st.mean(abs(r[f"{q}_{c}"] - r[f"{q}_ts1"]) for r in R)
    sil = [r for r in R if r["kind"] == "silence"]
    for q, pos in (("done", "success"), ("mis", "mistake")):
        P = [r for r in R if r["kind"] == pos]
        for c in ("ts1", "ts2", "ts05"): o[f"auc_{q}_{c}"] = auc([r[f"{q}_{c}"] for r in P], [r[f"{q}_{c}"] for r in sil])
    return o


res = dict(n_situations=len(K))
for name, _ in RUNS:
    R = [r for r in D[name] if key(r) in K]; est = stats(R); by = collections.defaultdict(list)
    for r in R: by[r["video_id"]].append(r)
    vids = sorted(by); rng = random.Random(0); B = collections.defaultdict(list)
    for _ in range(1000):
        S = [r for v in (rng.choice(vids) for _ in vids) for r in by[v]]
        for k, x in stats(S).items(): B[k].append(x)
    ci = {k: [round(sorted(v)[25], 3), round(sorted(v)[974], 3)] for k, v in B.items()}
    res[name] = {k: [round(v, 3), ci[k]] for k, v in est.items()}
    e = res[name]
    print(f"{name:30s} done shift x2 {e['done_ts2_shift']} x0.5 {e['done_ts05_shift']} | AUROC done {e['auc_done_ts1'][0]}/{e['auc_done_ts2'][0]}/{e['auc_done_ts05'][0]} "
          f"x2 CI {e['auc_done_ts2'][1]} | AUROC mis {e['auc_mis_ts1'][0]}/{e['auc_mis_ts2'][0]}/{e['auc_mis_ts05'][0]}")
json.dump(res, open(OUT + "/tp_analyze.json", "w"), indent=1)
