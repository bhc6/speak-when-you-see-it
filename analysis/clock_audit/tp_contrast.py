"""Two contrasts on the time-stamp probe (gpu/time_probe.py outputs, 411 shared validation situations; supplement H):
1. Direction: mean change of P(yes) on silent situations when every stamp is doubled and when it is halved. A model that read the
   stamps as time would move in opposite directions; equal signs mean it reacts to stamps unlike those of training.
2. ECT minus the variant without paired negatives in mistake AUROC (mistake vs silence) under each condition, with a 95% interval
   from 2000 bootstrap resamples of validation videos (seed 0).
python3 tp_contrast.py -> out/tp_contrast.json"""
import json, os, random, bisect, statistics as st, collections
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
SRC = V0 + "machine5/results/clk1122/"
RUNS = {"ECT s0": "tp_ECT_s0", "no paired neg. s0": "tp_noneg_s0", "plain fine-tuning": "tp_plain", "ECT recipe on black frames s0": "tp_blind_s0"}
key = lambda r: (r["video_id"], r["step"], r["kind"], r["t"])
D = {n: json.load(open(SRC + f + ".json")) for n, f in RUNS.items()}
K = set.intersection(*[{key(r) for r in d} for d in D.values()])
M = {n: {key(r): r for r in d if key(r) in K} for n, d in D.items()}


def auc(p, n):
    n = sorted(n); return sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p) / max(len(p) * len(n), 1)


res = {"n_situations": len(K), "silence_shift": {}, "ect_minus_noneg_mistake_auroc": {}}
for n in RUNS:
    S = [r for k, r in M[n].items() if k[2] == "silence"]
    res["silence_shift"][n] = {f"{q}_{c}": round(st.mean(r[f"{q}_{c}"] - r[f"{q}_ts1"] for r in S), 3) for q in ("done", "mis") for c in ("ts2", "ts05")}
a = lambda n, ks, c: auc([M[n][k][f"mis_{c}"] for k in ks if k[2] == "mistake"], [M[n][k][f"mis_{c}"] for k in ks if k[2] == "silence"])
by = collections.defaultdict(list)
for k in K: by[k[0]].append(k)
vids = sorted(by); rng = random.Random(0)
for c in ("ts1", "ts2", "ts05"):
    est = a("ECT s0", K, c) - a("no paired neg. s0", K, c); bs = []
    for _ in range(2000):
        ks = [k for v in (rng.choice(vids) for _ in vids) for k in by[v]]
        bs.append(a("ECT s0", ks, c) - a("no paired neg. s0", ks, c))
    bs.sort(); res["ect_minus_noneg_mistake_auroc"][c] = [round(est, 3), [round(bs[50], 3), round(bs[1949], 3)]]
print(json.dumps(res, indent=1))
json.dump(res, open(HERE + "/out/tp_contrast.json", "w"), indent=1)
