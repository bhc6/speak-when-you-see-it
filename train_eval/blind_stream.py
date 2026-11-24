"""Video-blind policies under the official STREAMING protocol (no GT restarts): step k starts when the policy itself
confirmed step k-1 (the first step at the video's first instruction time); checks every 5 s on the baseline grid; the
policy sees only the GT instruction text of its current step index, elapsed time since that start, and train priors.
Rules: time(A,B) and dur(c,B) as in blind_duration_envelope.py. Scored with the official eval; test-set-optimal points
(favourable to blind) + concave upper hull. python blind_stream.py EVAL_DIR DV SPLIT"""
import sys, json, statistics as st, collections
import types; _bs = types.ModuleType("bert_score"); _bs.BERTScorer = lambda *a, **k: None; sys.modules["bert_score"] = _bs  # BERT unused for timing metrics; keeps this CPU-only
EV, DV, SP = sys.argv[1:4]; sys.argv = [sys.argv[0], EV]
exec(open("WORKSPACE/qic/experiments/plan_repr/blind_baselines.py").read().split("val = utils.load_dataset")[0])
dur = collections.defaultdict(list); tr = utils.load_dataset("neurips2025", "advanced_planning", "train")
for a in tr.annotations:
    g = ev.get_ground_truth_data(tr, a["video_id"])
    for tx, ts in zip(g["texts"], g["texts_timestamps"]): dur[norm(tx[0])].append(max(ts) - ts[0])
gmed = st.median([d for v in dur.values() for d in v]); med = lambda s: st.median(dur[norm(s)]) if dur.get(norm(s)) else gmed
ds = utils.load_dataset(DV, "advanced_planning", SP); items = ds_items(ds)
def gen(pol):
    out = []
    for item in items:
        texts, ts = [], []; steps = item["steps"]; t0 = steps[0][1]; last = steps[0][2]; seek = int(t0 * 2)
        for k, (instr, _gstart, _) in enumerate(steps):
            kind, A, B = pol; A = A * med(instr) if kind == "dur" else A
            start = seek / 2; texts.append("Instruction: " + instr); ts.append(start); j = 1; fb = False; done = False
            while True:
                t = (seek + 10 * j) / 2; e = t - start
                if t > last: break
                if e >= A: texts.append("Success: Good job!"); ts.append(t); seek = seek + 10 * j; done = True; break
                if B is not None and not fb and e >= B: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
            if not done: break
        out.append({"video_id": item["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return out
pts = []
for A in range(10, 181, 10):
    for B in [None] + list(range(5, 181, 10)):
        if B is None or B < A: pts.append((("time", A, B), score(ds, gen(("time", A, B)))))
for c in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
    for B in (None, 15, 25): pts.append((("dur", c, B), score(ds, gen(("dur", c, B)))))
P = sorted({(s["ic"], s["F1"]) for _, s in pts}); h = []
for x in P:
    while len(h) >= 2 and (h[-1][0]-h[-2][0])*(x[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(x[0]-h[-2][0]) >= 0: h.pop()
    h.append(x)
best_f1 = max(pts, key=lambda p: (p[1]["F1"], p[1]["ic"])); best_ic = max(pts, key=lambda p: (p[1]["ic"], p[1]["F1"]))
json.dump({"points": [(p, s) for p, s in pts], "hull": h}, open(f"WORKSPACE/qic/results/blind_stream_{DV}_{SP}.json", "w"), default=str)
print(DV, SP, "F1-optimal", best_f1[0], best_f1[1]["ic"], best_f1[1]["F1"], "| IC-optimal", best_ic[0], best_ic[1]["ic"], best_ic[1]["F1"], "| hull", [(round(a,1), round(b,3)) for a, b in h])
