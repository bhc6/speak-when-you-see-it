import json, sys, os, random, statistics as st, collections
EV, DV, PF = sys.argv[1:4]; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ds = utils.load_dataset(DV, "advanced_planning", "test"); P = json.load(open(PF)); rng = random.Random(0)
pred_e, gt_e, early, late, inwin, nosucc = [], [], 0, 0, 0, 0; tps, fps = [], []; per_vid_ic = []
for p in P:
    g = ev.get_ground_truth_data(ds, p["video_id"]); T = list(zip(p["pred_texts"], p["pred_timestamps"]))
    segs, cur = [], None
    for x, t in T:
        if x.startswith("Instruction"): cur = []; segs.append(cur)
        elif cur is not None: cur.append((x, t))
    det = tot = 0
    for k, (tx, ts, ty, hm) in enumerate(zip(g["texts"], g["texts_timestamps"], g["texts_types"], g["has_mistake"])):
        start, end = ts[0], max(ts); s = segs[k] if k < len(segs) else []
        suc = [t for x, t in s if x.startswith("Success")]
        if not hm:
            tot += 1
            if suc:
                d = suc[0] - end; pred_e.append(suc[0] - start); gt_e.append(end - start)
                if abs(d) <= 15: inwin += 1
                elif d < 0: early += 1
                else: late += 1
                det += (abs(d) <= 15 and not any(x.startswith("Feedback") for x, _ in s))
            else: nosucc += 1
        gtm = [(x, t) for x, t, y in zip(tx[1:], ts[1:], ty[1:]) if "mistake" in y]
        for x, t in [(x, t) for x, t in s if x.startswith("Feedback")]:
            near = [m for m in gtm if abs(m[1] - t) <= 15]
            (tps if near else fps).append((p["video_id"], tx[0][:70], x[9:130], near[0][0][:110] if near else ""))
    if tot: per_vid_ic.append(det / tot)
n = len(pred_e); r = st.correlation(pred_e, gt_e) if n > 2 else float("nan")
import math
print(f"== {DV} {PF.split('/')[-2]}: clean steps with Success {n}, in ±15s {inwin}, early {early}, late {late}, no Success {nosucc}")
print(f"   corr(pred completion elapsed, GT step duration) = {r:.3f}  (blind fixed-delay rule = 0)")
print(f"   pred elapsed median {st.median(pred_e):.0f}s IQR [{sorted(pred_e)[n//4]:.0f},{sorted(pred_e)[3*n//4]:.0f}]  | GT duration median {st.median(gt_e):.0f}s")
print(f"   per-video IC: median {st.median(per_vid_ic):.2f}, videos with IC=0: {sum(v==0 for v in per_vid_ic)}/{len(per_vid_ic)}, top-5 share of detections ≈ see")
print(f"   feedback TP {len(tps)} FP {len(fps)}; distinct feedback texts {len({x for _,_,x,_ in tps+fps})}")
for lab, L in (("TP", tps), ("FP", fps)):
    for v, ins, fb, gt in rng.sample(L, min(int(os.environ.get("NS","6")), len(L))): print(f"   [{lab}] {v} | step: {ins} | said: {fb} | GT: {gt}")
