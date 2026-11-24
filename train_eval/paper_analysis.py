"""CPU-only analyses for the paper on EXISTING prediction files / dense logs (no new model runs, no test re-selection).
1. Per-mistake-category recall (official TP windows) on QICD test36 and Ego-MC for every frozen run present.
2. TP latency: predicted feedback time minus GT feedback time for TPs.
3. Validation IC-F1 frontier of T1v4@2400 (exact replay grid) next to the video-blind frontier (same split).
4. Calibration on validation (ECE, 10 bins) of first-token P(yes) for completion / mistake at GT-sampled situations
   (auroc57/auroc_r5 files): T1v4@2400 vs zero-shot.
python paper_analysis.py EVAL_DIR RESULTS_DIR OUT_JSON"""
import collections, contextlib, io, json, os, re, sys, glob
EV, RES, OUT = sys.argv[1:4]; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
out = {}
def cat_of(t): return t.split("mistake_")[-1].replace("_error", "") if "mistake" in t else None
for dv, sp, pat in (("neurips2025", "test", "*test36*"), ("ego_mc_bench", "test", "*_mc")):
    ds = utils.load_dataset(dv, "advanced_planning", sp)
    for d in sorted(glob.glob(f"{RES}/**/{pat}/pred.json", recursive=True)):
        P = json.load(open(d)); name = d.split("/")[-2] + "@" + d.split("/")[-3]
        rec = collections.Counter(); tot = collections.Counter(); lat = []
        for p in P:
            g = ev.get_ground_truth_data(ds, p["video_id"])
            fb = [(t, x) for x, t in zip(p["pred_texts"], p["pred_timestamps"]) if x.startswith("Feedback")]
            for k, (tx, ts, ty) in enumerate(zip(g["texts"], g["texts_timestamps"], g["texts_types"])):
                for x, t, y in zip(tx[1:], ts[1:], ty[1:]):
                    if "mistake" not in y: continue
                    c = cat_of(y); tot[c] += 1
                    near = [pt for pt, _ in fb if abs(pt - t) <= 15.0]
                    if near: rec[c] += 1; lat.append(min(near, key=lambda pt: abs(pt - t)) - t)
        out[f"{dv}/{name}"] = {"recall_by_category": {c: [rec[c], tot[c], round(rec[c] / tot[c], 3)] for c in sorted(tot)},
                               "tp_latency_median_s": sorted(lat)[len(lat) // 2] if lat else None, "n_tp_any": len(lat)}
g = [json.loads(l) for l in open(f"{RES}/dense/gridall_T1v4_s2400.jsonl")] if os.path.exists(f"{RES}/dense/gridall_T1v4_s2400.jsonl") else []
front = lambda rows: sorted({(r["ic"], r["F1"]) for r in rows if not any(o["ic"] >= r["ic"] and o["F1"] > r["F1"] or o["ic"] > r["ic"] and o["F1"] >= r["F1"] for o in rows)})
out["val_frontier_T1v4_s2400"] = front([r for r in g if r.get("caps", 0) == 0])
bf = f"{RES}/blind_frontier_neurips2025_validation.json"
if os.path.exists(bf): out["val_frontier_blind"] = front(json.load(open(bf)))
def ece(ps, ys, nb=10):
    B = collections.defaultdict(list)
    for p, y in zip(ps, ys): B[min(int(p * nb), nb - 1)].append((p, y))
    return round(sum(len(v) / len(ps) * abs(sum(p for p, _ in v) / len(v) - sum(y for _, y in v) / len(v)) for v in B.values()), 4)
for name, f in (("zero", "auroc57/zero.json"), ("T1v4_s2400", "auroc_r5/T1v4_s2400.json")):
    f = f"{RES}/{f}"
    if not os.path.exists(f): continue
    R = json.load(open(f))
    out[f"ece_{name}"] = {"done": ece([r["p_done"] for r in R], [r["kind"] == "success" for r in R]),
                          "mistake": ece([r["p_mis"] for r in R], [r["kind"] == "mistake" for r in R])}
json.dump(out, open(OUT, "w"), indent=1); print(json.dumps(out)[:3000])
