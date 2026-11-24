"""Video-blind rule families for GuideMe (Anchor-Based protocol, GT history). A rule sees no pixels, only
(domain, activity name, #prior GT turns k, seconds since the last GT turn e) and cross-fitted priors
(5 video folds: GuideMe-Train annotations are not public, so the other test folds stand in for a training set).

Every family speaks when k == 0 ('no assistant turn yet' -> give the first instruction).
  never / always
  late(th2)            speak iff e >= th2                        fixed delay since last turn (QICD 'fixed delay')
  prior(alpha)         speak iff e >= alpha * G(k)               G = cross-fitted median gap before turn k for this
                                                                 activity (>=3 videos) else domain (QICD duration prior)
  window(th1, th2)     speak iff e < th1 or e >= th2             th1 > 0 speaks right after a turn; th1 <= 5 is exactly
                                                                 the 'silent instances are >= 5 s from any anchor' gap
  prior_window(th1,a)  speak iff e < th1 or e >= alpha * G(k)
  hazard(tau)          speak iff P_hat(anchor | domain, k-bin, e-bin) >= tau, P_hat = smoothed counts on other folds
  dom_<fam>            a separate parameter per domain (domain is evident from the request text)
Frontier 'Q' (QICD-isomorphic) = never, always, late, prior, dom_late, dom_prior.
Frontier 'F' (full)            = Q + window, prior_window, dom_window, hazard.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

EBINS = [0, 1, 2, 3, 4, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 10 ** 9]
KBINS = [0, 1, 2, 4, 8, 16, 10 ** 9]
TH2 = list(range(0, 121)) + list(range(125, 401, 5)) + list(range(450, 2001, 50)) + [10 ** 9]
TH1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 45, 60, 90]
ALPHA = [float(a) for a in np.round(np.concatenate([np.arange(0, 3.01, 0.05), np.arange(3.25, 10.01, 0.25)]), 3)] + [1e9]
TAU = [float(t) for t in np.round(np.arange(0, 1.0001, 0.01), 3)]
QFAM = ["never", "always", "late", "prior", "dom_late", "dom_prior"]
FFAM = QFAM + ["window", "prior_window", "dom_window", "hazard"]


def _bin(x, edges):
    return int(np.searchsorted(edges, x, side="right") - 1)


def gap_prior(videos, inst):
    lists = defaultdict(list)
    for v in videos:
        g = np.diff(v["ts"]).tolist()
        if v["act"]:
            lists[(v["fold"], "act", v["act"])].append(g)
        lists[(v["fold"], "dom", v["dom"])].append(g)
    cache = {}

    def G(fold, act, dom, k):
        ck = (fold, act, dom, k)
        if ck not in cache:
            val = None
            for key in ([("act", act)] if act else []) + [("dom", dom)]:
                ls = [g for (f, a, b), gl in lists.items() if f != fold and (a, b) == key for g in gl]
                if len(ls) < 3:
                    continue
                atk = [g[k - 1] for g in ls if len(g) >= k]
                val = float(np.median(atk)) if len(atk) >= 3 else float(np.median([x for g in ls for x in g]))
                break
            cache[ck] = val
        return cache[ck]

    return np.array([G(x["fold"], x["act"], x["dom"], x["k"]) if x["k"] else 0.0 for x in inst])


def hazard_prior(inst, nfold=5):
    key = [(x["dom"], _bin(x["k"], KBINS), _bin(x["e"], EBINS)) for x in inst]
    key2 = [(x["dom"], _bin(x["e"], EBINS)) for x in inst]
    y = np.array([x["is_anchor"] for x in inst], float)
    fold = np.array([x["fold"] for x in inst])
    out = np.zeros(len(inst))
    for f in range(nfold):
        tr = fold != f
        c1, n1, c2, n2 = (defaultdict(float) for _ in range(4))
        for i in np.where(tr)[0]:
            c1[key[i]] += y[i]; n1[key[i]] += 1; c2[key2[i]] += y[i]; n2[key2[i]] += 1
        base = y[tr].mean()
        for i in np.where(~tr)[0]:
            p2 = (c2[key2[i]] + 2 * base) / (n2[key2[i]] + 2)
            out[i] = (c1[key[i]] + 5 * p2) / (n1[key[i]] + 5)
    return out


class Feats:
    def __init__(self, videos, inst, doms):
        self.k = np.array([x["k"] for x in inst])
        self.e = np.array([x["e"] for x in inst], float)
        self.y = np.array([x["is_anchor"] for x in inst])
        fold = np.array([x["fold"] for x in inst])
        dom = np.array([doms.index(x["dom"]) for x in inst])
        self.group = dom * 5 + fold  # 20 (domain, fold) groups
        self.G = gap_prior(videos, inst)
        self.H = hazard_prior(inst)


def base_rules(F: Feats):
    """{family: [(params, mask)]} for the non-per-domain families."""
    first = F.k == 0
    e, G = F.e, F.G
    R = {"never": [({}, np.zeros_like(first))], "always": [({}, np.ones_like(first))]}
    R["late"] = [({"th2": t}, first | (e >= t)) for t in TH2]
    R["prior"] = [({"alpha": a}, first | (e >= a * G)) for a in ALPHA]
    R["window"] = [({"th1": t1, "th2": t2}, first | (e < t1) | (e >= t2)) for t1 in TH1[1:] for t2 in TH2]
    R["prior_window"] = [({"th1": t1, "alpha": a}, first | (e < t1) | (e >= a * G)) for t1 in TH1[1:] for a in ALPHA]
    R["hazard"] = [({"tau": t}, first | (F.H >= t)) for t in TAU]
    return R


def table(F: Feats, rules):
    """counts[r, g, (CS, FA, NR, PC)] for every rule r and (domain, fold) group g."""
    y = F.y
    out = np.zeros((len(rules), 20, 4), np.int64)
    for i, (_, m) in enumerate(rules):
        for c, sel in enumerate([~m & ~y, m & ~y, ~m & y, m & y]):
            out[i, :, c] = np.bincount(F.group[sel], minlength=20)
    return out
