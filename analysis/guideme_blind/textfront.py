"""Temporal-alignment (sP / sR / sF1, official Stage 2) for video-blind rules on GuideMe-Test (Anchor + GT history).

A blind rule must also say something when it speaks. Text sources (no pixels, no future GT):
  generic  'Good job. Now move on to the next step.'
  retr     retrieval prior: the other-fold video whose request is nearest (all-mpnet-base-v2 cosine) -> its
           assistant utterance number min(k, n-1)                     (cross-fitted 'training-set' script)
  echo     repeat the last GT assistant utterance in the dialogue history (k >= 1; k == 0 -> retr)
Timing rules: a grid from every family plus all frontier vertices of seed 0 (see frontier.py).
Selection: test-selected = best mean sF1 over (rule, text) on the test set; CV = chosen on 4 video folds, applied to the 5th.

  .venv/bin/python textfront.py [seed ...]     -> results/text_frontier_seed{S}.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import common as C
import frontier as FR
import rules as R

GENERIC = "Good job. Now move on to the next step."


def per_video_alignment(samples, sts):
    sil = {"silent", "<|silent|>"}
    out = {}
    for s in samples:
        outs = []
        for c in s["conversations"]:
            if "pred" not in c:
                continue
            gen = "" if c["pred"].lower().strip() in sil else c["pred"]
            ref = "" if c["value"].lower().strip() in sil else c["value"]
            outs.append(C.ss.FrameOutput(gen=gen, ref=ref, frame_idx_in_stream=c["timestamp"],
                                         timestamp_in_stream=float(c["timestamp"])))
        mr, _ = C.ss.find_match_by_segments(outs, sts_model=sts, gaussian_sigma=0.01, batch_size=64, debug=False)
        m = C.ss.compute_precision_recall_f1(mr, semantic_score_threshold=0.5, fps=1)
        out[s["video_id"]] = (m["soft_precision"], m["soft_recall"], m["soft_f1"])
    return out


def main(seeds):
    V = C.load_videos()
    sts = C.CachedSTS(C.HERE / "data" / "emb_cache.npz")
    sts.warm([t for v in V for t in v["txt"]] + [GENERIC] + [v["req"] for v in V])
    req = np.stack([sts.emb[v["req"]] for v in V])
    req = req / np.linalg.norm(req, axis=1, keepdims=True)
    sim = req @ req.T
    nn = {}
    for i, v in enumerate(V):
        cand = [j for j, w in enumerate(V) if w["fold"] != v["fold"]]
        nn[v["vid"]] = V[max(cand, key=lambda j: sim[i, j])]
    for seed in seeds:
        res, (inst, F, fam_rules, points, cv, mask_of) = FR.run_seed(V, seed, verify=False)
        texts = {
            "generic": [GENERIC] * len(inst),
            "retr": [nn[x["vid"]]["txt"][min(x["k"], len(nn[x["vid"]]["txt"]) - 1)] for x in inst],
        }
        texts["echo"] = [x["last_txt"] if x["k"] else r for x, r in zip(inst, texts["retr"])]
        cands = {}
        for p in [p for _, _, p in FR.upper_hull([q for q in points if q[2]["rule"] in R.FFAM])] + \
                 [p for _, _, p in FR.upper_hull([q for q in points if q[2]["rule"] in R.QFAM])]:
            cands[json.dumps({k: v for k, v in p.items() if k not in ("CS", "FA", "NR", "PC", "params_by_domain")},
                             sort_keys=True)] = p
        for f, grid in [("late", [0, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 200, 400, 10 ** 9]),
                        ("prior", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])]:
            for i, (par, _) in enumerate(fam_rules[f]):
                if list(par.values())[0] in grid:
                    p = {"rule": f, **par, "choice": [i] * 20}
                    cands[json.dumps(p, sort_keys=True)] = p
        for i, (par, _) in enumerate(fam_rules["window"]):
            if par["th1"] in (3, 5, 8, 12) and par["th2"] in (10, 15, 20, 30, 45, 60, 90, 120, 200, 400, 10 ** 9):
                p = {"rule": "window", **par, "choice": [i] * 20}
                cands[json.dumps(p, sort_keys=True)] = p
        for i, (par, _) in enumerate(fam_rules["hazard"]):
            if round(par["tau"] * 100) % 5 == 0:
                p = {"rule": "hazard", **par, "choice": [i] * 20}
                cands[json.dumps(p, sort_keys=True)] = p
        print(f"seed {seed}: {len(cands)} timing rules x {len(texts)} texts", flush=True)
        rows = []
        t0 = time.time()
        for key, p in cands.items():
            m = mask_of(p)
            fam = "Q" if p["rule"] in R.QFAM else "F"
            for tn, tx in texts.items():
                preds = [tx[i] if m[i] else "Silent" for i in range(len(inst))]
                pv = per_video_alignment(C.to_samples(inst, preds, V), sts)
                b = C.behavior(C.to_samples(inst, ["x" if s else "Silent" for s in m], V))
                rows.append({"rule": key, "family_set": fam, "text": tn, "FA_rate": 100 - b["TNR"], "TPR": b["TPR"],
                             "per_video": pv,
                             **{n: 100 * float(np.mean([v[j] for v in pv.values()])) for j, n in enumerate(["sP", "sR", "sF1"])}})
            print(f"  {len(rows)} done, {time.time() - t0:.0f}s; last {p['rule']} best so far sF1 "
                  f"{max(r['sF1'] for r in rows):.1f}", flush=True)
        fold = {v["vid"]: v["fold"] for v in V}
        summ = {}
        for fs, allowed in [("Q", ("Q",)), ("F", ("Q", "F"))]:
            rr = [r for r in rows if r["family_set"] in allowed]
            best = max(rr, key=lambda r: r["sF1"])
            cvv = []
            for fo in range(5):
                trm = lambda r: np.mean([v[2] for vid, v in r["per_video"].items() if fold[vid] != fo])
                b = max(rr, key=trm)
                cvv += [v for vid, v in b["per_video"].items() if fold[vid] == fo]
            summ[fs] = {"test_best": {k: best[k] for k in ("rule", "text", "sP", "sR", "sF1", "FA_rate", "TPR")},
                        "cv": {n: 100 * float(np.mean([v[j] for v in cvv])) for j, n in enumerate(["sP", "sR", "sF1"])},
                        "best_by_text": {tn: max((r["sF1"] for r in rr if r["text"] == tn), default=None) for tn in texts}}
        print(json.dumps(summ, indent=1), flush=True)
        for r in rows:
            r.pop("per_video")
        json.dump({"seed": seed, "summary": summ, "rows": rows}, open(FR.OUT / f"text_frontier_seed{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0])
