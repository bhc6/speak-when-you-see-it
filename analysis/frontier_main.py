"""QICD MAIN test (109 videos) video-blind frontier, same construction as frontier_canon.py (upper hull of single blind rules,
realized hull-edge mixtures and realized rule mixtures, tuned on the target test set), plus margins of published Main
per-step results (values as reported by their papers / the challenge leaderboard; not re-run). python frontier_main.py DIR"""
import json, sys
D = sys.argv[1] if len(sys.argv) > 1 else "machine4/results"
env = json.load(open(f"{D}/blind_envelope_neurips2025_main_test.json")); mixf = json.load(open(f"{D}/blind_mixture_neurips2025_main_test.json"))
rules = json.load(open(f"{D}/blind_frontier_neurips2025_main_test.json"))
pts = [(h[0], h[1]) for h in env["hull_vertices"]] + [(m[3], m[4]) for m in env.get("hull_mixtures", [])] + [(m[1], m[2]) for m in mixf["mix"]] + \
      [(r["ic"], r["F1"]) for r in rules]
pts = sorted(set(pts)); h = []
for x in pts:
    while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (x[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) >= 0: h.pop()
    h.append(x)
top = max(h, key=lambda x: x[1]); h = [p for p in h if p[0] >= top[0]]
def fr(ic):
    best = 0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in h if i >= ic])
PUB = [("LiveMamba (QICD paper)", 51.0, 0.19), ("Qwen2.5-VL-7B (QICD paper)", 38.9, 0.06), ("MR-CAS (CVPR26 challenge)", 56.2, 0.31),
       ("Second entry (CVPR26 challenge)", 31.4, 0.20), ("Qwen3-VL-8B baseline (challenge)", 19.8, 0.00), ("Qwen2-VL-7B (QICD paper)", 19.4, 0.11),
       ("VideoLLaMA3-7B (QICD paper)", 17.8, 0.15)]
res = dict(hull=[(round(a, 1), round(b, 3)) for a, b in h], max_blind_ic=max(r["ic"] for r in rules), max_blind_f1=max(r["F1"] for r in rules),
           dur_val_selected=env.get("dur_val_selected"), published=[(n, ic, f, round(fr(ic), 3), round(f - fr(ic), 3)) for n, ic, f in PUB])
json.dump(res, open("audit_1114/frontier_main.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
