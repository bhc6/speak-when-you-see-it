"""Pareto frontier (IC vs F1) of video-blind policies on a split -- deliberately evaluated ON THE TARGET SPLIT (gives the
blind baselines their best possible parameters: conservative for any comparison). python blind_frontier.py EVAL_DIR DV SPLIT"""
import sys, json
EV, DV, SP = sys.argv[1:4]; sys.argv = [sys.argv[0], EV]
src = open("WORKSPACE/qic/experiments/plan_repr/blind_baselines.py").read().split("val = utils.load_dataset")[0]
exec(src)
ds = utils.load_dataset(DV, PS if DV == "neurips2025" else "advanced_planning", SP)
rows = []
for A in list(range(10, 181, 10)) + [240, 300]:
    for B in [None] + list(range(5, 181, 10)):
        if B is not None and B >= A: continue
        for r in ([None] if B is None else ([None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8] if DV == "neurips2025" else [None])):
            rows.append(dict(A=A, B=B, r=r, **score(ds, preds(ds, A, B, r))))
for ic_min in (0, 20, 30, 40, 42, 45, 50, 60, 67, 70):
    c = [x for x in rows if x["ic"] >= ic_min]
    if c: b = max(c, key=lambda x: x["F1"]); print(f"IC>={ic_min:2d}: best blind F1 {b['F1']:.3f} (A={b['A']},B={b['B']},r={b['r']}, IC {b['ic']}, P {b['P']}, R {b['R']})")
m = max(rows, key=lambda x: x["ic"]); print(f"max blind IC {m['ic']} (A={m['A']},B={m['B']}) F1 {m['F1']}")
json.dump(rows, open(f"WORKSPACE/qic/results/blind_frontier_{DV}{TAG}_{SP}.json", "w"))
