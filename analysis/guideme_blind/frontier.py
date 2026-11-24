"""Response-behaviour (CS/NR/FA/PC) blind frontier on GuideMe-Test, Anchor-Based protocol, GT history.

  .venv/bin/python frontier.py            -> results/behavior_frontier.json

Plane: x = FA rate on silent instances (100 - TNR), y = PC rate on anchors (TPR). Mixtures of rules (per-instance hash
coin) reach the upper concave hull in expectation, so the frontier is the ROC hull of all rule points.
Test-selected = all rule parameters chosen on the test set itself (most favourable to the blind side).
CV-selected   = for each objective max(CS + lam*PC), each family's parameter chosen on 4 video folds, applied to the
                5th, counts summed over folds (no test-set selection).
Every hull vertex is re-scored with the official evaluate_sample() and must match the vectorised counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common as C
import rules as R

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
LAMS = [float(x) for x in np.round(np.geomspace(0.02, 50, 140), 4)]  # objective CS + lam * PC (counts)
PROTOCOL_GAP = C.SILENT_GAP

# Tab. 1 of arXiv 2607.02991 (main results, full test set, Anchor + GT history): CS, NR, FA, PC, Score, sP, sR, sF1
PUBLISHED = {
    "Doubao-Seed-1.8": (8.8, 6.1, 34.8, 44.2, 38.8, 30.6, 47.5, 36.8),
    "GPT-5.2": (13.4, 16.8, 30.6, 39.2, 38.9, 30.2, 36.8, 32.5),
    "Gemini 3 Pro": (17.5, 22.0, 19.0, 41.5, 44.8, 39.3, 36.9, 36.5),
    "Gemini 3.1 Pro": (7.4, 6.3, 36.4, 49.9, 37.8, 29.3, 46.9, 35.7),
    "Qwen2.5-VL-7B": (19.7, 20.4, 25.4, 34.5, 37.5, 30.1, 30.9, 29.3),
    "Qwen3-VL-8B": (5.2, 6.7, 37.7, 50.5, 31.9, 29.7, 46.8, 35.7),
    "Qwen3-VL-30B-A3B": (6.7, 6.0, 36.5, 50.9, 33.3, 29.5, 46.9, 35.6),
    "Qwen3.5-397B-A17B": (35.2, 37.9, 7.9, 19.0, 45.7, 38.4, 20.2, 21.7),
    "VideoLLM-online": (0.0, 0.0, 43.6, 56.4, 22.6, 22.3, 41.1, 28.7),
    "Dispider": (43.6, 56.3, 0.0, 0.1, 43.6, 1.0, 0.1, 0.1),
    "LiveStar": (21.2, 24.4, 22.2, 32.3, 25.0, 20.7, 19.0, 19.1),
    "MMDuet2": (41.7, 57.9, 0.2, 0.2, 41.8, 3.3, 0.2, 0.4),
    "Qwen3-VL-8B (FT on GuideMe-Train)": (19.5, 20.6, 24.5, 35.3, 42.1, 40.3, 43.7, 39.7),
}


def roc(c):
    s, a = c["CS"] + c["FA"], c["PC"] + c["NR"]
    return 100 * c["FA"] / s, 100 * c["PC"] / a


def upper_hull(pts):
    """pts: list of (x, y, payload). Upper-left concave hull from (0, y0) to (100, 100)."""
    P = sorted(set((round(x, 9), round(y, 9)) for x, y, _ in pts))
    pay = {(round(x, 9), round(y, 9)): p for x, y, p in pts}
    H = []
    for p in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) >= 0:
            H.pop()
        H.append(p)
    # keep only the monotone-increasing part (upper-left frontier)
    F, best = [], -1
    for p in H:
        if p[1] > best:
            F.append(p); best = p[1]
    return [(x, y, pay[(x, y)]) for x, y in F]


def hull_y(H, x):
    xs, ys = [h[0] for h in H], [h[1] for h in H]
    return float(np.interp(x, xs, ys))


def score_bound(H, sil_share, q):
    """max over the hull of Score = CS% + q * PC% (q = mean judge score/100 of the spoken anchors)."""
    return max(sil_share * (100 - x) / 100 + q * (100 - sil_share) * y / 100 for x, y, _ in H)


def pt(counts):
    c = dict(zip(["CS", "FA", "NR", "PC"], (int(v) for v in counts)))
    x, y = roc(c)
    return x, y, c


def select(T, groups, lam):
    """index of the rule maximising CS + lam * PC summed over `groups`."""
    v = T[:, groups, :].sum(1)
    return int(np.argmax(v[:, 0] + lam * v[:, 3]))


def run_seed(V, seed, gap=C.SILENT_GAP, verify=True):
    C.SILENT_GAP = gap
    inst = C.instances(V, seed)
    C.SILENT_GAP = PROTOCOL_GAP
    F = R.Feats(V, inst, C.DOMS)
    sil_share = 100 * (~F.y).sum() / len(F.y)
    RB = R.base_rules(F)
    pool = {"dom_late": RB["late"], "dom_prior": RB["prior"], "dom_window": RB["late"] + RB["window"]}
    fam_rules = {**RB, **pool}
    T = {f: R.table(F, rs) for f, rs in fam_rules.items()}
    G = np.arange(20)
    dom_of, fold_of = G // 5, G % 5
    points = []  # (x, y, payload) ; payload carries 'choice' = rule index per group
    for f, rs in RB.items():
        for i, (par, _) in enumerate(rs):
            x, y, c = pt(T[f][i].sum(0))
            points.append((x, y, {"rule": f, **par, **c, "choice": [i] * 20}))
    for f in pool:
        for lam in LAMS:
            ch = [0] * 20
            for d in range(4):
                gi = G[dom_of == d]
                j = select(T[f], gi, lam)
                for g in gi:
                    ch[g] = j
            x, y, c = pt(sum(T[f][ch[g], g] for g in G))
            points.append((x, y, {"rule": f, "lam": lam, **c, "choice": ch,
                                  "params_by_domain": {C.DOMS[d]: pool[f][ch[d * 5]][0] for d in range(4)}}))
    cv = []
    for f in fam_rules:
        for lam in LAMS:
            ch = [0] * 20
            for fo in range(5):
                for d in (range(4) if f.startswith("dom_") else [None]):
                    tr = G[(fold_of != fo) & ((dom_of == d) if d is not None else True)]
                    te = G[(fold_of == fo) & ((dom_of == d) if d is not None else True)]
                    j = select(T[f], tr, lam)
                    for g in te:
                        ch[g] = j
            x, y, c = pt(sum(T[f][ch[g], g] for g in G))
            cv.append((x, y, {"rule": f, "lam": lam, **c, "choice": ch}))
    res = {"seed": seed, "gap": gap, "n": len(inst), "silent_share": sil_share}

    def mask_of(p):
        m = np.zeros(len(F.y), bool)
        for g in G:
            sel = F.group == g
            m[sel] = fam_rules[p["rule"]][p["choice"][g]][1][sel]
        return m

    for name, fams in [("Q", R.QFAM), ("F", R.FFAM)]:
        for kind, pts in [("test", points), ("cv", cv)]:
            H = upper_hull([p for p in pts if p[2]["rule"] in fams])
            if verify:
                for x, y, p in H:  # official evaluate_sample on every vertex
                    m = mask_of(p)
                    b = C.behavior(C.to_samples(inst, ["Next step." if s else "Silent" for s in m], V))
                    assert abs(b["TNR"] - (100 - x)) < 1e-6 and abs(b["TPR"] - y) < 1e-6, (p["rule"], b, x, y)
            res[f"hull_{name}_{kind}"] = [{"FA_rate": x, "TPR": y, **{k: v for k, v in p.items() if k != "choice"}}
                                          for x, y, p in H]
    res["family_hulls_test"] = {f: [{"FA_rate": x, "TPR": y} for x, y, _ in upper_hull(
        [p for p in points if p[2]["rule"] in (f, "never", "always")])] for f in fam_rules}

    return res, (inst, F, fam_rules, points, cv, mask_of)


def compare(res):
    rows = []
    H = lambda r, key: [(h["FA_rate"], h["TPR"], h) for h in r[key]]
    for name, (cs, nr, fa, pc, score, sp, sr, sf1) in PUBLISHED.items():
        tnr, tpr = 100 * cs / (cs + fa), 100 * pc / (pc + nr)
        q = (score - cs) / pc if pc > 0.5 else None
        row = {"model": name, "CS": cs, "NR": nr, "FA": fa, "PC": pc, "Score": score, "sF1": sf1,
               "FA_rate": 100 - tnr, "TPR": tpr, "implied_q": q}
        for fr in ["Q", "F"]:
            for kind in ["test", "cv"]:
                ys = [hull_y(H(r, f"hull_{fr}_{kind}"), 100 - tnr) for r in res]
                row[f"blindTPR_{fr}_{kind}"] = float(np.mean(ys))
                row[f"blindTPR_{fr}_{kind}_range"] = [float(min(ys)), float(max(ys))]
                if q is not None:
                    row[f"blindScore_{fr}_{kind}"] = float(np.mean(
                        [score_bound(H(r, f"hull_{fr}_{kind}"), r["silent_share"], q) for r in res]))
        rows.append(row)
    return rows


def main():
    V = C.load_videos()
    res = []
    for s in SEEDS:
        r, _ = run_seed(V, s)
        r.pop("_masks", None)
        res.append(r)
        print(f"seed {s}: silent {r['silent_share']:.2f}  " + "  ".join(
            f"{k}:{len(r[k])}" for k in r if k.startswith("hull_")), flush=True)
    rows = compare(res)
    for row in rows:
        print(f"{row['model']:<34} FA {row['FA_rate']:5.1f} TPR {row['TPR']:5.1f} | Q {row['blindTPR_Q_test']:5.1f}/"
              f"{row['blindTPR_Q_cv']:5.1f}  F {row['blindTPR_F_test']:5.1f}/{row['blindTPR_F_cv']:5.1f}"
              + (f" | q {row['implied_q']:.2f} Score {row['Score']} blind Q {row['blindScore_Q_test']:.1f}/"
                 f"{row['blindScore_Q_cv']:.1f} F {row['blindScore_F_test']:.1f}/{row['blindScore_F_cv']:.1f}"
                 if row["implied_q"] is not None else ""))
    # protocol counterfactual: silent instances >= g s from anchors (paper uses 5)
    cf = {}
    for g in [1, 3, 5, 10]:
        r, _ = run_seed(V, 0, gap=g, verify=False)
        cf[g] = {"silent_share": r["silent_share"],
                 **{f"TPR_at_FA{x}_{fr}": hull_y([(h["FA_rate"], h["TPR"], h) for h in r[f"hull_{fr}_test"]], x)
                    for fr in ["Q", "F"] for x in [5, 20, 50]}}
        print("gap", g, {k: round(v, 1) for k, v in cf[g].items()}, flush=True)
    json.dump({"seeds": res, "published": rows, "gap_counterfactual": cf}, open(OUT / "behavior_frontier.json", "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
