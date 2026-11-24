"""Deterministic MIXTURES of video-blind policies (per-step hash h(video,step) in [0,1): steps with h < q use blind policy
a, the rest blind policy b). Extends the blind frontier to its upper envelope on the target split (favourable to the
blind side). python blind_mixture.py EVAL_DIR DV SPLIT"""
import sys, json, hashlib
EV, DV, SP = sys.argv[1:4]; sys.argv = [sys.argv[0], EV]
exec(open("WORKSPACE/qic/experiments/plan_repr/blind_baselines.py").read().split("val = utils.load_dataset")[0])
ds = utils.load_dataset(DV, PS if DV == "neurips2025" else "advanced_planning", SP)
items = ds_items(ds)
def hh(v, k): return int(hashlib.md5(f"{v}/{k}".encode()).hexdigest(), 16) / 16 ** 32
def mixpreds(pa, pb, q):
    out = []
    for item in items:
        texts, ts = [], []
        for k, (instr, start, last) in enumerate(item["steps"]):
            A, B, r = pa if hh(item["video_id"], k) < q else pb
            texts.append("Instruction: " + instr); ts.append(start)
            init = int(start * 2); j = 1; fb = False; want = B is not None and (r is None or ((mrate(instr) or 0) >= r))
            while True:
                t = (init + 10 * j) / 2; e = t - start
                if t > last: break
                if e >= A: texts.append("Success: Good job!"); ts.append(t); break
                if want and not fb and e >= B: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
        out.append({"video_id": item["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return out
rows = json.load(open(f"WORKSPACE/qic/results/blind_frontier_{DV}{TAG}_{SP}.json"))
bf = max(rows, key=lambda x: x["F1"]); bi = max(rows, key=lambda x: (x["ic"], x["F1"]))
pa, pb = (bi["A"], bi["B"], bi["r"]), (bf["A"], bf["B"], bf["r"])
res = []
for q in [i / 20 for i in range(21)]:
    s = score(ds, mixpreds(pa, pb, q)); res.append((q, s["ic"], s["F1"])); print(f"q={q:.2f} IC {s['ic']} F1 {s['F1']}", flush=True)
json.dump({"ic_policy": pa, "f1_policy": pb, "mix": res}, open(f"WORKSPACE/qic/results/blind_mixture_{DV}{TAG}_{SP}.json", "w"))
