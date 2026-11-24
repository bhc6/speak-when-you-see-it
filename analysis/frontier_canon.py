"""Canonical video-blind frontier (the one used for the published-results re-read, refcheck/published_frontier_egomc.json):
upper convex hull of single blind rules (hull_vertices), realized hull-edge mixtures, and realized rule mixtures, all on the
target test set. fr(ds, ic) = best blind F1 at IC-Acc >= ic. Margin = F1 - fr."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "data" / "frontier"
def hull(dv):
    d = json.load(open(f"{R}/blind_envelope_{dv}_test.json"))
    pts = [(h[0], h[1]) for h in d["hull_vertices"]] + [(m[3], m[4]) for m in d.get("hull_mixtures", [])] + \
          [(m[1], m[2]) for m in json.load(open(f"{R}/blind_mixture_{dv}_test.json"))["mix"]]
    pts = sorted(set(pts)); h = []
    for x in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (x[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) >= 0: h.pop()
        h.append(x)
    return h
H = {"qicd": hull("neurips2025"), "mc": hull("ego_mc_bench")}
def fr(ds, ic):
    h = H[ds]; best = 0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b: best = max(best, fa + (fb - fa) * (ic - a) / max(b - a, 1e-9))
    return max([best] + [f for i, f in h if i >= ic])
if __name__ == "__main__":
    for k, v in H.items(): print(k, [(round(a, 1), round(b, 3)) for a, b in v])
