"""Is the mistake F1 of a run on Ego-MC-Bench explained by how often it raises alarms, or by which steps / moments it picks?
Re-implements the official mistake matching (per-step segments; +-15 s strict; each GT mistake counted once; repeats inside a window
free; every feedback on a mistake-free step is a FP) and validates it against the tp/fp/fn of the official eval.log. Then compares
the actual F1 with two nulls that keep the number of alarms of the run:
  time-shuffled: same steps, same count per step, alarm times drawn uniformly from the 5-s checks of that step;
  step-shuffled: the per-step alarm patterns (offsets from the step start) permuted across the steps of the same video."""
import json, random, sys, statistics as st
import os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "DATA_ROOT"; A = {a["video_id"]: a for a in json.load(open(f"{R}/paper_assets/egomc_gt/annotations.json"))}   # public annotations of Ego-MC-Bench (annotations/annotations.json of the dataset repository)
def gt_segments(a):
    segs = []
    for tx, ty, ts in zip(a["texts"], a["text_types"], a["text_timestamps"]):
        if ty == "Instruction": segs.append(dict(start=ts, mist=[], last=ts))
        elif segs:
            segs[-1]["last"] = max(segs[-1]["last"], ts)
            if ty == "Feedback": segs[-1]["mist"].append(ts)
    for i, s in enumerate(segs): s["next"] = segs[i + 1]["start"] if i + 1 < len(segs) else s["last"] + 15
    return segs
def pred_segments(p):
    segs = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruction"): segs.append(dict(fb=[], succ=None))
        elif segs and t.startswith("Feedback"): segs[-1]["fb"].append(ts)
        elif segs and t.startswith("Success") and segs[-1]["succ"] is None: segs[-1]["succ"] = ts
    return segs
def score(G, F):   # G: gt segments, F: list of feedback-time lists per predicted segment
    tp = fp = 0; ngt = sum(len(g["mist"]) for g in G)
    for i, fb in enumerate(F):
        if i >= len(G) or not G[i]["mist"]: fp += len(fb); continue
        reg = set()
        for t in fb:
            j = min(range(len(G[i]["mist"])), key=lambda k: abs(G[i]["mist"][k] - t))
            if abs(G[i]["mist"][j] - t) < 15.0: reg.add(j)
            else: fp += 1
        tp += len(reg)
    return tp, fp, ngt - tp
def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
RUNS = [("No paired neg., s0", "machine4/results/jobs/ABL_none_greedy_mc", (148, 324, 249)), ("No paired neg., s1", "machine5/results/jobs/ABL_none_s1_greedy_mc", (131, 258, 266)),
        ("No neg., re-balanced", "machine4/results/jobs/T1v2_2400_greedy_mc", (120, 238, 277)), ("+ notdone, s0", "machine8/results/jobs/ABL_noCL_greedy_mc", (82, 170, 315)),
        ("+ clean, s0", "machine0/results/jobs/ABL_noND_greedy_mc", (63, 107, 334)), ("ECT, s0", "machine0/results/final/sft_greedy_mc", (52, 50, 345)), ("ECT, s1", "machine4/results/seeds/final/s1_greedy_mc", (67, 125, 330)),
        ("ECT, s2", "machine5/results/seeds/final/s2_greedy_mc", (77, 80, 320)), ("ECT, s3", "machine5/results/jobs/s3_greedy_mc", (83, 133, 314)),
        ("Qwen3-VL-2B zero-shot (memory)", "machine0/results/final/r1_mem_mc", (92, 367, 305)), ("Qwen3.8-Max zero-shot", "machine11/results/jobs/API_q38max_mc", (128, 266, 269))]
N = 500; rng = random.Random(0); out = {}
print(f"{'run':32s} official tp/fp/fn   ours tp/fp/fn    P     R     F1   | time-shuffled F1 [95%]      step-shuffled F1 [95%]     alarms on mistake steps")
for name, d, off in RUNS:
    P = json.load(open(f"{R}/{d}/pred.json")); tot = [0, 0, 0]; vids = []
    for p in P:
        G = gt_segments(A[p["video_id"]]); S = pred_segments(p); F = [s["fb"] for s in S]; vids.append((G, S, F))
        for k, v in enumerate(score(G, F)): tot[k] += v
    nulls = {"time": [], "step": []}
    for _ in range(N):
        a = [0, 0, 0]; b = [0, 0, 0]
        for G, S, F in vids:
            Ft = []
            for i, s in enumerate(S):
                if i >= len(G) or not s["fb"]: Ft.append([]); continue
                hi = s["succ"] if s["succ"] is not None else max(G[i]["next"], max(s["fb"]))
                grid = [G[i]["start"] + 5 * k for k in range(1, max(1, int((hi - G[i]["start"]) // 5)) + 1)]
                Ft.append(rng.sample(grid, min(len(s["fb"]), len(grid))) + [rng.choice(grid) for _ in range(max(0, len(s["fb"]) - len(grid)))])
            for k, v in enumerate(score(G, Ft)): a[k] += v
            n = min(len(G), len(S)); off_ = [[t - G[i]["start"] for t in S[i]["fb"]] for i in range(n)]; perm = list(range(n)); rng.shuffle(perm)
            Fs = [[G[i]["start"] + o for o in off_[perm[i]]] for i in range(n)] + F[n:]
            for k, v in enumerate(score(G, Fs)): b[k] += v
        nulls["time"].append(f1(*a)); nulls["step"].append(f1(*b))
    ci = lambda v: (sorted(v)[int(.025 * N)], sorted(v)[int(.975 * N)])
    tp, fp, fn = tot; Pr = tp / (tp + fp); Rc = tp / (tp + fn); F = f1(*tot)
    nfb = sum(len(f) for _, _, FF in vids for f in FF); onm = sum(len(f) for G, _, FF in vids for i, f in enumerate(FF) if i < len(G) and G[i]["mist"])
    t, s = nulls["time"], nulls["step"]
    print(f"{name:32s} {str(off):17s} {str(tuple(tot)):16s} {Pr:.3f} {Rc:.3f} {F:.3f} | {st.mean(t):.3f} [{ci(t)[0]:.3f}, {ci(t)[1]:.3f}]  p={sum(x >= F for x in t) / N:.3f}   {st.mean(s):.3f} [{ci(s)[0]:.3f}, {ci(s)[1]:.3f}]  p={sum(x >= F for x in s) / N:.3f}   {onm}/{nfb} = {onm / nfb:.2f}")
    out[name] = dict(official=off, ours=tot, P=round(Pr, 3), R=round(Rc, 3), F1=round(F, 3), time_shuffled=[round(st.mean(t), 3), round(ci(t)[0], 3), round(ci(t)[1], 3)], step_shuffled=[round(st.mean(s), 3), round(ci(s)[0], 3), round(ci(s)[1], 3)],
                     gain_over_time=round(F - st.mean(t), 3), gain_over_step=round(F - st.mean(s), 3), alarms=nfb, alarms_on_mistake_steps=onm)
ng = sum(len(g["mist"]) for a in A.values() for g in gt_segments(a)); ns = sum(len(gt_segments(a)) for a in A.values()); nm = sum(bool(g["mist"]) for a in A.values() for g in gt_segments(a))
print("GT: steps", ns, "with a mistake", nm, f"({nm / ns:.2f})", "mistakes", ng)
json.dump(out, open(f"{HERE}/supp/f1_null.json", "w"), indent=1)
rows = "\n".join(f"    {n.replace('s0', 'seed 0').replace('s1', 'seed 1').replace('s2', 'seed 2').replace('s3', 'seed 3').replace(' (memory)', '')} & {o['alarms']} & {o['alarms_on_mistake_steps'] / o['alarms']:.2f} & {o['P']:.3f} & {o['R']:.3f} & {o['F1']:.3f} & {o['time_shuffled'][0]:.3f} [{o['time_shuffled'][1]:.3f}, {o['time_shuffled'][2]:.3f}] & {o['step_shuffled'][0]:.3f} [{o['step_shuffled'][1]:.3f}, {o['step_shuffled'][2]:.3f}] \\\\" for n, o in out.items())
open(f"{HERE}/supp/tab_f1null.tex", "w").write(r"""% generated by analysis_f1_null.py -- do not edit
\begin{table*}[t]
  \centering
  \small
  \setlength{\tabcolsep}{5pt}
  \resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%
  \begin{tabular}{@{}lccccccc@{}}
    \toprule
    Run (Ego-MC-Bench, greedy decoding) & Alarms & On mistake steps & P & R & F1 & F1, times shuffled [95\%] & F1, steps shuffled [95\%] \\
    \midrule
""" + rows + r"""
    \bottomrule
  \end{tabular}}
  \caption{What the mistake F1 on Ego-MC-Bench rewards. ``On mistake steps'': share of the alarms raised in steps that contain a mistake (53\% of the steps do). ``Times shuffled'': the alarms stay in their steps and are moved to random checks of the step. ``Steps shuffled'': the alarm patterns are permuted across the steps of the same video. Means and 95\% ranges over 500 draws; our re-implementation of the official matching reproduces the counts of the official evaluation for every run.}
  \label{tab:f1null}
\end{table*}
""")
