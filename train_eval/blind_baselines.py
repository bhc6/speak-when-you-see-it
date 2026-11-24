"""Video-blind control policies under the official turn-based protocol + official eval (no model, no video).
Each step k: checks at the baseline's grid t = (int(start*2) + 10 j) / 2, j = 1, 2, ...; the policy sees only the GT
instruction text, the GT step start (both given by the protocol) and elapsed time.
  time     : complete when elapsed >= A; one feedback per step when elapsed >= B (B=None: never)
  prior    : like time, but feedback only for instructions whose TRAIN mistake rate >= r (instruction-text prior)
Parameters are selected on validation (max F1, ties IC), then applied ONCE to test / Ego-MC. Feedback text is a constant
(F1/P/R/IC depend on timing only).
python blind_baselines.py EVAL_DIR"""
import contextlib, io, json, os, re, sys, collections
EV = sys.argv[1]; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
from datasets import load_dataset
HF = "qualcomm/qualcomm-interactive-cooking-dataset"
PS = os.environ.get("QIC_PLAN_SET", "advanced_planning"); TAG = "" if PS == "advanced_planning" else "_" + PS   # plan set (QICD main: QIC_PLAN_SET=main)

def norm(s): return " ".join(s.strip().lower().rstrip(".").split())
rate = collections.defaultdict(lambda: [0, 0])
for r in load_dataset(HF, PS, split="train").to_list():
    cur = None
    for x, y in zip(r["output_texts"], r["output_types"]):
        if y == "instruction": cur = norm(x); rate[cur][1] += 1
        elif "mistake" in y and cur is not None and rate[cur][0] < rate[cur][1]: rate[cur][0] += 1
mrate = lambda s: (rate[norm(s)][0] / rate[norm(s)][1]) if rate[norm(s)][1] else None

def preds(ds, A, B, r=None):
    out = []
    for item in ds_items(ds):
        texts, ts = [], []
        for instr, start, last in item["steps"]:
            texts.append("Instruction: " + instr); ts.append(start)
            init = int(start * 2); j = 1; fb = False
            want_fb = B is not None and (r is None or ((mrate(instr) or 0) >= r))
            while True:
                t = (init + 10 * j) / 2; e = t - start
                if t > last: break
                if e >= A: texts.append("Success: Good job!"); ts.append(t); break
                if want_fb and not fb and e >= B: texts.append("Feedback: you made a mistake."); ts.append(t); fb = True
                j += 1
        out.append({"video_id": item["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return out

def ds_items(ds):
    items = []
    vids = [a["video_id"] for a in ds.annotations]
    for v in vids:
        g = ev.get_ground_truth_data(ds, v)
        end_all = max(max(t) for t in g["texts_timestamps"]) + 120
        items.append({"video_id": v, "steps": [(tx[0], ts[0], end_all) for tx, ts in zip(g["texts"], g["texts_timestamps"])]})
    return items

def score(ds, P):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(ds, P)
    o = buf.getvalue(); ic = float(re.findall(r"IC-Acc: ([\d.]+)", o)[-1])
    tp, fp, tn, fn = map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1])
    p = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
    return dict(ic=ic, tp=tp, fp=fp, tn=tn, fn=fn, P=round(p, 3), R=round(rr, 3), F1=round(2 * p * rr / max(p + rr, 1e-9), 3))

val = utils.load_dataset("neurips2025", "advanced_planning", "validation")
grid = []
for A in range(10, 181, 10):
    for B in [None] + list(range(5, 181, 10)):
        if B is not None and B >= A: continue
        for r in ([None] if B is None else [None, 0.2, 0.4, 0.6]):
            grid.append(((A, B, r), score(val, preds(val, A, B, r))))
for kind, filt in (("time", lambda k: k[2] is None), ("prior", lambda k: True)):
    sel = max((g for g in grid if filt(g[0])), key=lambda g: (g[1]["F1"], g[1]["ic"]))
    best_ic = max((g for g in grid if filt(g[0])), key=lambda g: (g[1]["ic"], g[1]["F1"]))
    print(f"[{kind}] val selected (A,B,r)={sel[0]} {sel[1]} | val max-IC {best_ic[0]} {best_ic[1]}")
    A, B, r = sel[0]
    print(f"[{kind}] QICD test36 : {score(utils.load_dataset('neurips2025', 'advanced_planning', 'test'), preds(utils.load_dataset('neurips2025', 'advanced_planning', 'test'), A, B, r))}")
    mc = utils.load_dataset("ego_mc_bench", "advanced_planning", "test")
    print(f"[{kind}] Ego-MC (time part only): {score(mc, preds(mc, A, B, None))}")
