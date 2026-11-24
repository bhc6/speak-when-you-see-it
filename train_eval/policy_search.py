"""Grid search of alert policies on dense scores with recipe-grouped 5-fold cross-validation.

python policy_search.py DENSE_JSON OUT_JSON   (cwd = official eval repo, PYTHONPATH=./:<analysis dir>)

For every (done_rule, mis_rule, cap) in the pre-registered grid (DESIGN_alert_policy.md) the official counts are
computed once per video (eval.run on a one-video list; counts are additive over videos except IC/P/R/F1, which are
ratios of sums), so any fold / bootstrap aggregate is exact. Selection: in each fold pick the config maximising the
objective on the 4 training folds, record its counts on the held-out fold; out-of-fold totals = sum over folds.
Objectives: F1 (primary) and IC (secondary). Baselines B0 (text/text), B1 (text, cap 1), B2 (text/never) are always
reported on all videos and out-of-fold.
"""

from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool

import policy_sim as ps

T = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DONE = ["text"] + [f"thr:{t}" for t in T] + [f"k:{k}:{t}" for k in (2, 3) for t in (0.5, 0.7)]
MIS = (["text", "never"] + [f"thr:{t}" for t in T] + [f"k:{k}:{t}" for k in range(2, 7) for t in (0.5, 0.7, 0.9)]
       + [f"cusum:{d}:{h}" for d in (0, 0.5, 1, 2) for h in (1, 2, 4, 8)])
CAPS = [1, 10**9]
BASE = {"B0": ("text", "text", 10**9), "B1": ("text", "text", 1), "B2": ("text", "never", 10**9)}


def per_video_counts(dense, cfg):
    return ps.score_pervideo(ps.simulate(dense, *cfg))


_DENSE = None


def _worker(cfg):
    return per_video_counts(_DENSE, cfg)


def main(dense_path, out_path):
    dense = ps.load_dense(dense_path)
    den = ps.ic_denominators(dense)
    vids = [v["video_id"] for v in dense]
    recipe = {v: v.split("_")[0] for v in vids}
    recipes = sorted(set(recipe.values()), key=int)
    folds = {r: i % 5 for i, r in enumerate(recipes)}
    cfgs = list(BASE.values()) + [c for c in itertools.product(DONE, MIS, CAPS) if c[1] != "never" or c[2] > 1]
    cfgs = list(dict.fromkeys(cfgs))
    global _DENSE
    _DENSE = dense
    table = {}
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as pool:
        for i, (cfg, pv) in enumerate(zip(cfgs, pool.imap(_worker, cfgs, chunksize=4))):
            assert all(pv[v][1] == den[v] for v in vids)
            table["|".join(map(str, cfg))] = pv
            if i % 50 == 0: print(f"{i}/{len(cfgs)}", flush=True)
    json.dump({"folds": {v: folds[recipe[v]] for v in vids}, "table": table}, open(out_path, "w"))
    print("saved", out_path, len(table), "configs")


if __name__ == "__main__":
    main(*sys.argv[1:3])
