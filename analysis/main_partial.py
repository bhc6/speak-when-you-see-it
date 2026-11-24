"""Interim (NOT a result): official eval restricted to the videos finished so far. python partial.py EVAL_DIR shard*.json"""
import sys, json, os, io, re, contextlib
EV = sys.argv[1]; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
P = [p for f in sys.argv[2:] for p in json.load(open(f))]; vids = {p["video_id"] for p in P}
ds = utils.load_dataset("neurips2025", os.environ.get("PS", "main"), "test"); ds.annotations = [a for a in ds.annotations if a["video_id"] in vids]
buf = io.StringIO()
with contextlib.redirect_stdout(buf): ev.run(ds, P)
o = buf.getvalue(); ic = re.findall(r"IC-Acc: ([\d.]+)", o)[-1]
tp, fp, tn, fn = map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1])
p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
print(f"videos {len(vids)} IC {ic} TP/FP/FN {tp}/{fp}/{fn} P {p:.3f} R {r:.3f} F1 {2*p*r/max(p+r,1e-9):.3f}")
