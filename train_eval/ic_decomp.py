"""Decompose why each clean (no-GT-mistake) step does or does not count for IC-Acc, following eval.py's order of
checks, plus FP timing and example feedbacks. python ic_decomp.py RUN_JSON [...] (cwd eval repo, PYTHONPATH=./)"""
import json, sys, random, collections
from utils import load_dataset
from eval import check_if_correct_instruction_given

ds = load_dataset("neurips2025", "advanced_planning", "validation")
GT = {ds[i]["video_id"]: ds[i] for i in range(len(ds))}


def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruction"): out.append([])
        if out: out[-1].append((t, ts))
    return out


for path in sys.argv[1:]:
    preds = json.load(open(path)); c = collections.Counter(); fp_t = []; ex = []
    for p in preds:
        g = GT[p["video_id"]]; S = segs(p)
        for i in range(len(g["texts"])):
            if g["has_mistake"][i]:
                continue
            if i >= len(S): c["not reached (earlier step never done)"] += 1; continue
            seg = S[i]; end = max(g["texts_timestamps"][i]); start = g["texts_timestamps"][i][0]
            if len(seg) == 1: c["no output in step"] += 1; continue
            if not check_if_correct_instruction_given(g["texts"][i][0], seg[0][0].replace("Instruct:", "").strip()):
                c["instruction mismatch"] += 1; continue
            fbs = [(t, ts) for t, ts in seg if t.startswith("Feedback")]
            for t, ts in fbs:
                fp_t.append(ts - start)
                if len(ex) < 400: ex.append((p["video_id"], g["texts"][i][0][:60], round(ts - start), t[10:170]))
            last = seg[-1]
            if fbs:
                c["has feedback -> lost"] += 1
            elif not last[0].startswith("Success"):
                c["silent but never Success"] += 1
            elif abs(last[1] - end) < 15:
                c["HIT"] += 1
            elif last[1] < end:
                c["Success too early"] += 1
            else:
                c["Success too late"] += 1
    n = sum(c.values())
    print(f"\n### {path}: clean steps {n}")
    for k, v in c.most_common(): print(f"   {k:40s} {v:4d} ({v / n:.0%})")
    print(f"   FP in clean steps at first check (<=5 s): {sum(x <= 5.01 for x in fp_t) / len(fp_t):.0%}; within first 15 s: {sum(x <= 15 for x in fp_t) / len(fp_t):.0%}")
    random.seed(0)
    for e in random.sample(ex, 6): print("   EX", e)
