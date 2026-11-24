"""Time-matched evidence pairs: pair accuracy = P(p_pos > p_neg) (+0.5 per tie) per question type; any elapsed-time-only
policy scores exactly 0.5. 95% CI by video bootstrap (2000). Also 'both-correct at 0.5' (pos>0.5 and neg<0.5, MMVP-style).
python epair_summary.py DIR_WITH_JSONS..."""
import json, glob, os, sys, random, collections
rows = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*.json")):
        k = os.path.basename(f)[:-5]; x = json.load(open(f))
        if k not in rows or len(x) > len(rows[k]): rows[k] = x
def acc(L): return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in L) / max(1, len(L))
out = {}
for k, x in sorted(rows.items()):
    r = {"n": len(x)}
    for q, tag in (("instruction_end", "done"), ("mistake_inference", "mis")):
        P = [(e["video_id"], e["p_pos"], e["p_neg"]) for e in x if e["q"] == q]
        byv = collections.defaultdict(list)
        for v, a, b in P: byv[v].append((a, b))
        vids = sorted(byv); rng = random.Random(0); bs = []
        for _ in range(2000):
            s = [p for v in (rng.choice(vids) for _ in vids) for p in byv[v]]; bs.append(acc(s))
        bs.sort(); r[f"{tag}_n"] = len(P); r[f"{tag}_pairacc"] = round(acc([(a, b) for _, a, b in P]), 3)
        r[f"{tag}_ci"] = (round(bs[50], 3), round(bs[1949], 3))
        r[f"{tag}_both"] = round(sum(a > .5 and b < .5 for _, a, b in P) / max(1, len(P)), 3)
    out[k] = r
json.dump(out, open("DATA_ROOT/audit_1114/epair_summary.json", "w"), indent=1)
for k, r in out.items():
    print(f"{k:16s} n={r['n']:4d} | done {r['done_pairacc']} {r['done_ci']} both {r['done_both']} | mis {r['mis_pairacc']} {r['mis_ci']} both {r['mis_both']}")
