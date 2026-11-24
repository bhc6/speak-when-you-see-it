"""Failure-mode analysis of finished official-protocol runs (no GPU). Run from the official eval repo dir with
PYTHONPATH=./ so data.py's loader is used. python failure_modes.py RUN_JSON [RUN_JSON ...]"""
import json, re, sys, collections, statistics as st
from utils import load_dataset

ds = load_dataset("neurips2025", "advanced_planning", "validation")
GT = {ds[i]["video_id"]: ds[i] for i in range(len(ds))}
W = 15.0


def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruction"): out.append([])
        if out: out[-1].append((t, ts))
    return out


KW = {
    "divergent(not following / instead / unrelated)": r"not following|instead of|instead|unrelated|different (step|action|task)|not (the )?(correct|current) step|does not (align|match)|not aligned|deviat",
    "not yet done / incomplete": r"not (yet )?(completed|finished|done)|has not|haven't|have not|still|incomplete|partially|about to|yet to",
    "measurement": r"teaspoon|tablespoon|cup|amount|measur|quantity|too much|too little|\d/\d",
    "tool/utensil": r"knife|spoon|bowl|pan|plate|board|utensil|tool|lid|towel",
    "hygiene/safety": r"wash|clean|hygien|safe|glove|dirty|spill",
    "temperature/time": r"heat|temperatur|minute|second|microwave|stove|boil",
}


def analyze(path):
    preds = json.load(open(path))
    st_ = collections.Counter(); fb_rep = []; first_fb = []; fp_kw = collections.Counter(); tp_kw = collections.Counter()
    succ_err = []; never_done = 0; total_steps = 0; reached = 0; gt_steps = 0; dup = 0; n_fb = 0
    mis_steps_detect = collections.Counter(); types = collections.Counter(); types_hit = collections.Counter()
    for p in preds:
        g = GT[p["video_id"]]; S = segs(p); gt_steps += len(g["texts"]); reached += min(len(S), len(g["texts"]))
        for i, seg in enumerate(S[:len(g["texts"])]):
            total_steps += 1
            gts, gtt, gty = g["texts"][i], g["texts_timestamps"][i], g["texts_types"][i]
            end = max(gtt); has = g["has_mistake"][i]
            fbs = [(t, ts) for t, ts in seg if t.startswith("Feedback")]
            sus = [ts for t, ts in seg if t.startswith("Success")]
            n_fb += len(fbs)
            if sus: succ_err.append(sus[-1] - end)
            else: never_done += 1
            if fbs:
                fb_rep.append(len(fbs)); first_fb.append(fbs[0][1] - gtt[0])
                texts = [t.lower() for t, _ in fbs]; dup += len(texts) - len(set(texts))
            mts = [(ts, ty) for ts, ty in zip(gtt, gty) if "mistake" in ty]
            for ts, ty in mts:
                key = ("unaligned" if "unaligned" in ty else "aligned") + ":" + re.sub(r".*mistake_", "", ty)
                types[key] += 1
                if any(abs(fts - ts) < W for _, fts in fbs): types_hit[key] += 1
            for t, fts in fbs:
                hit = has and any(abs(fts - ts) < W for ts, _ in mts)
                tl = t.lower()
                for k, rx in KW.items():
                    if re.search(rx, tl): (tp_kw if hit else fp_kw)[k] += 1
                st_["TPfb" if hit else ("FP_noGT" if not has else "FP_offwindow")] += 1
            st_["clean_step" if not has else "mistake_step"] += 1
            if not has and not fbs: st_["clean_silent"] += 1
    nfp = st_["FP_noGT"] + st_["FP_offwindow"]; ntp = st_["TPfb"]
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else float("nan")
    print(f"\n### {path}")
    print(f"steps reached {reached}/{gt_steps} ({reached / gt_steps:.0%}); steps never judged done {never_done}/{total_steps} ({never_done / max(total_steps,1):.0%})")
    print(f"Success timing (pred - GT end) n={len(succ_err)}: median {st.median(succ_err):+.0f}s, |err|<15s {sum(abs(x)<15 for x in succ_err)/len(succ_err):.0%}, early(<-15s) {sum(x<-15 for x in succ_err)/len(succ_err):.0%}, late(>15s) {sum(x>15 for x in succ_err)/len(succ_err):.0%}")
    print(f"feedback lines {n_fb}: TP-lines {ntp}, FP in clean steps {st_['FP_noGT']}, FP in mistake steps but off-window {st_['FP_offwindow']}; exact-duplicate texts {dup}")
    print(f"steps with >=1 feedback: {len(fb_rep)}; feedback per such step median {st.median(fb_rep):.0f} p90 {q(fb_rep,.9)}; first feedback after step start median {st.median(first_fb):.0f}s (5 s = first possible check)")
    print(f"clean steps {st_['clean_step']}, of which silent {st_['clean_silent']} ({st_['clean_silent']/st_['clean_step']:.0%})")
    print("recall by GT mistake type (hit/total):", ", ".join(f"{k} {types_hit[k]}/{v}" for k, v in types.most_common()))
    print("feedback content (share of FP lines | share of TP lines):")
    for k in KW: print(f"   {k:48s} {fp_kw[k]/max(nfp,1):5.0%} | {tp_kw[k]/max(ntp,1):5.0%}")


for p in sys.argv[1:]: analyze(p)
