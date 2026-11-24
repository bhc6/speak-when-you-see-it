"""Qualitative examples for supplement G (success and failure cases), chosen by a fixed rule before any frame is looked at.
Run: ECT seed 0, official greedy decoding, QICD test36 (results/final/sft_greedy_test36, 41.2 / 0.217).
Per-step outcome by the official eval.py logic (same segmentation, instruction check and 15-s window):
  A on-time confirmation : mistake-free step, exactly one Success, within 15 s of the step end
  B timely alarm         : mistake step, a Feedback within 15 s of an annotated mistake (TP)
  C false alarm          : mistake-free step with a Feedback
  D missed mistake       : mistake step without any Feedback
  E mistimed confirmation: mistake-free step, no Feedback, Success missing or >= 15 s from the step end
Rule: in each class sort the candidates by annotated step duration (ties by video id, step) and take the lower median.
Reported alongside on the same step: zero-shot Qwen3-VL-2B (memory prompt, results/test_vl_mem.json) and the same recipe
without paired negatives (ABL_none seed 0); cls None = the official parser does not align that step (instruction check fails).
Decision time t (the frames are the model window at t): A/E the Success (E without Success: the first check after the
step end + 15 s), B/C the Feedback, D the first official check at or after the annotated mistake.
python select_examples.py OUT_JSON"""
import sys, os, json, math
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"
RUNS = {"ect_s0": B + "results/final/sft_greedy_test36/pred.json", "zeroshot": B + "results/test_vl_mem.json",
        "no_neg": B + "results/jobs/ABL_none_greedy_test36/pred.json"}
W = 15.0
ds = L.load("neurips2025", "test")
for name, p in RUNS.items():  # the runs must reproduce their official numbers before anything is selected
    r, _ = L.run(ds, json.load(open(p))); print(name, r)

def outcomes(pred):
    out = {}
    for item in pred:
        v = item["video_id"]; g = L.ev.get_ground_truth_data(ds, v)
        segs, segt = L.ev.segment_pred_feedbacks(item["pred_texts"], item["pred_timestamps"])
        for k in range(len(g["texts"])):
            texts, ts, ty, hm = g["texts"][k], g["texts_timestamps"][k], g["texts_types"][k], g["has_mistake"][k]
            st, en = ts[0], max(ts); mis = [(t, x) for t, y, x in zip(ts, ty, texts) if "mistake" in y and x is not None]
            P = list(zip(segs[k][1:], segt[k][1:])) if k < len(segs) else []
            ok = k < len(segs) and L.ev.check_if_correct_instruction_given(texts[0], segs[k][0].replace("Instruct:", "").strip())
            fb = [(x, t) for x, t in P if "Feedback" in x and "all the steps" not in x]; sc = [t for x, t in P if "Success:" in x]
            o = dict(video=v, step=k, instruction=texts[0], start=st, end=en, dur=en - st, has_mistake=bool(hm), ok=ok,
                     gt_mistakes=[{"t": t, "text": x} for t, x in mis], feedback=[{"t": t, "text": x} for x, t in fb], success=sc)
            if not ok: o["cls"] = None
            elif not hm:
                if fb: o["cls"] = "C"
                elif len(P) == 1 and sc and abs(sc[0] - en) < W: o["cls"] = "A"
                else: o["cls"] = "E"
            else:
                hit = [f for f in fb if mis and min(abs(f[1] - t) for t, _ in mis) < W]
                o["cls"] = "B" if hit else ("D" if not fb else None)   # alarms only at wrong times: neither class
                if hit: o["hit"] = {"t": hit[0][1], "text": hit[0][0]}
            out[(v, k)] = o
    return out

E = outcomes(json.load(open(RUNS["ect_s0"]))); Z = outcomes(json.load(open(RUNS["zeroshot"]))); N = outcomes(json.load(open(RUNS["no_neg"])))
def check_time(st, want):
    init = int(st * 2); j = max(1, math.ceil((want * 2 - init) / 10 - 1e-9)); return (init + 10 * j) / 2
sel = {}
for c in "ABCDE":
    cand = sorted([o for o in E.values() if o["cls"] == c], key=lambda o: (o["dur"], o["video"], o["step"]))
    o = dict(cand[(len(cand) - 1) // 2]); z = Z[(o["video"], o["step"])]; nn = N[(o["video"], o["step"])]
    if c in "A": t = o["success"][0]
    elif c == "E": t = o["success"][0] if o["success"] else check_time(o["start"], o["end"] + W)
    elif c == "B": t = o["hit"]["t"]
    elif c == "C": t = o["feedback"][0]["t"]
    else: t = check_time(o["start"], o["gt_mistakes"][0]["t"])
    o.update(n_candidates=len(cand), decision_t=t, zeroshot={"cls": z["cls"], "feedback": z["feedback"], "success": z["success"]},
             no_neg={"cls": nn["cls"], "feedback": nn["feedback"], "success": nn["success"]})
    sel[c] = o; print(c, len(cand), o["video"], o["step"], round(o["dur"], 1), o["instruction"][:60], "| zs:", z["cls"], "| no-neg:", nn["cls"])
json.dump({"rule": __doc__, "counts": {c: sum(o["cls"] == c for o in E.values()) for c in "ABCDE"}, "selected": sel},
          open(sys.argv[1], "w"), indent=1)
