"""Replay the official turn-based baseline from a qic_dense.py log under threshold decisions and score with the
official eval (qualcomm_interactive_cooking_eval/eval.py; timing-based TP/FP so placeholder feedback text is exact
for IC/P/R/F1, not for BERT/ROUGE).
order done_first   : completion question first (baseline order); feedback only if not completed.
order mistake_first: mistake question first; if it fires the step continues, otherwise completion by its threshold.
EXACT ONLY IF caps == 0 (and tau_done <= 0.9). A replay that needs a check beyond the logged cap ('cap') is NOT
exact and not one-directional: the real baseline keeps checking the step (extra FP feedbacks) or completes it later
and continues with later steps. Every output row carries exact=(caps==0 and the step-end guards below); rank only
exact rows; confirm the chosen config with a real qic_prob.py run.
python dense_sim.py --dense A.jsonl [B.jsonl ...] --eval_dir DIR [--split validation] [--grid] [--tau_done X --tau_mis Y --order O --out PRED.json]"""
import argparse, contextlib, io, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dense", nargs="+", required=True); ap.add_argument("--eval_dir", required=True)
ap.add_argument("--split", default="validation"); ap.add_argument("--dataset_version", default="neurips2025")
ap.add_argument("--grid", action="store_true"); ap.add_argument("--tau_done", type=float, default=0.5)
ap.add_argument("--tau_mis", type=float, default=0.5); ap.add_argument("--order", default="mistake_first"); ap.add_argument("--out")
ap.add_argument("--score", default="lm", choices=["lm", "sia"], help="replay LM first-token probs (cols 2,3) or SIA type probs (cols 4,5)")
a = ap.parse_args()
sys.path.insert(0, a.eval_dir); os.chdir(a.eval_dir)
import eval as ev, utils  # noqa: E402
ev.get_bert_score = lambda *x: 0.0   # placeholder feedback text: BERT is meaningless here and slow; IC/P/R/F1 unaffected
DS = utils.load_dataset(a.dataset_version, "advanced_planning", a.split)
V = [json.loads(l) for f in a.dense for l in open(f) if l.strip()]
_ids = [v["video_id"] for v in V]
assert len(_ids) == len(set(_ids)), "duplicate videos in dense log"
assert os.environ.get("DS_SUBSET_OK") or len(_ids) == len(DS), f"dense log covers {len(_ids)} videos, split has {len(DS)}"


def replay(td, tm, order):
    lim = min((v.get("stop_sia") if a.score == "sia" and v.get("stop_sia") is not None else v.get("stop", 0.9)) for v in V)
    if td > lim + 1e-9: return None, None      # not exact: tau_done above the logged stop level
    preds, caps = [], 0
    for v in V:
        texts, ts = [], []
        for s in v["steps"]:
            texts.append("Instruction: " + s["instruction"]); ts.append(s["start"])
            fin = False; last = None
            for c in s["checks"]:
                seek, t = c[0], c[1]; pd, pm = (c[4], c[5]) if a.score == "sia" else (c[2], c[3])
                last = seek
                if order == "mistake_first":
                    if pm >= tm: texts.append("Feedback: you made a mistake."); ts.append(t); continue
                    if pd >= td: texts.append("Success: Good job!"); ts.append(t); fin = True; break
                else:
                    if pd >= td: texts.append("Success: Good job!"); ts.append(t); fin = True; break
                    if pm >= tm: texts.append("Feedback: you made a mistake."); ts.append(t)
            if not fin:
                caps += s["why"] in ("cap", "stop")   # 'stop' w/o completion: ran off the log (rounding edge case)
                break          # never completed -> the baseline runs to the end of the video
            if last + 10 >= v["n_frames"]: break
        preds.append({"video_id": v["video_id"], "pred_texts": texts, "pred_timestamps": ts})
    return preds, caps


def score(preds):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.run(DS, preds)
    o = buf.getvalue()
    # the official eval prints cumulative metrics after every video: the last print is the full-set result
    ic = float(re.findall(r"IC-Acc: ([\d.]+)", o)[-1])
    tp, fp, tn, fn = map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1])
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(ic=ic, tp=tp, fp=fp, tn=tn, fn=fn, P=round(p, 3), R=round(r, 3), F1=round(2 * p * r / max(p + r, 1e-9), 3))


print(f"{len(V)} videos", file=sys.stderr)
if os.environ.get("GRID2"):   # done_first is exact for any tau_mis (the step completes at or before a logged stop)
    for td in (0.1, 0.2, 0.3, 0.4, 0.5):
        for tm in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
            pr, caps = replay(td, tm, "done_first")
            if pr is None: continue
            print(json.dumps(dict(order="done_first", tau_done=td, tau_mis=tm, caps=caps, exact=caps == 0, **score(pr))), flush=True)
    sys.exit(0)
if a.grid:
    for order in ("done_first", "mistake_first"):
        for td in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for tm in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
                pr, caps = replay(td, tm, order)
                if pr is None: continue
                print(json.dumps(dict(order=order, tau_done=td, tau_mis=tm, caps=caps, exact=caps == 0, **score(pr))), flush=True)
else:
    pr, caps = replay(a.tau_done, a.tau_mis, a.order)
    print(json.dumps(dict(order=a.order, tau_done=a.tau_done, tau_mis=a.tau_mis, caps=caps, **score(pr))))
    if a.out: json.dump(pr, open(a.out, "w"))
