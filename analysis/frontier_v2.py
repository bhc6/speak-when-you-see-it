"""Video-blind frontier of the paper: upper convex hull of the measured rule policies (frontier_canon.py)
together with the learned blind policy on the paper's inputs ('strict' in data/blind_strong/summary_strong.json:
instruction text, step index, elapsed time, training priors; tuned on the test set like the rules). fr(ds, ic) = best blind F1
at IC-Acc >= ic; margin = F1 - fr."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import frontier_canon as F
S = json.load(open(HERE.parent / "data" / "blind_strong" / "summary_strong.json"))["hulls"]


def hull(pts):
    pts = sorted(set((round(a, 4), round(b, 4)) for a, b in pts)); h = []
    for x in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (x[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) >= 0: h.pop()
        h.append(x)
    best = max(b for _, b in h); i = max(k for k, (_, b) in enumerate(h) if b == best)   # keep the decreasing part
    return [(0.0, best)] + [p for p in h[i:] if p[0] > 0]


H = {ds: hull(list(F.H.get(ds, [])) + [tuple(p) for p in S[ds]["strict"]]) for ds in ("qicd", "mc", "main")}


def fr(ds, ic):
    h = H[ds]; best = 0
    for (a, fa), (b, fb) in zip(h, h[1:]):
        if a <= ic <= b and b > a: best = max(best, fa + (fb - fa) * (ic - a) / (b - a))
    return max([best] + [f for i, f in h if i >= ic])


def margin(ds, ic, f1): return round(f1 - fr(ds, ic), 3)


if __name__ == "__main__":
    for ds, h in H.items(): print(ds, "max IC", h[-1][0], [(round(a, 1), round(b, 3)) for a, b in h])
