"""Fidelity check: replay the baseline decision rule (text/text) on dense scores and compare, step by step, with a
real official-protocol run of the same model/prompt/decoding. Turn-based steps restart at their GT start, so the two
must agree exactly up to the dense horizon (GT end + margin). Usage: python fidelity.py DENSE_GLOB REAL_JSON"""
import glob, json, sys
import policy_sim as ps

dense = [v for f in sorted(glob.glob(sys.argv[1])) for v in json.load(open(f))]
real = {p["video_id"]: p for p in json.load(open(sys.argv[2]))}
sim = {p["video_id"]: p for p in ps.simulate(dense, "text", "text")}
hor = {v["video_id"]: [s["gt_end"] + 30 for s in v["steps"]] for v in dense}


def segs(p):
    out = []
    for t, ts in zip(p["pred_texts"], p["pred_timestamps"]):
        if t.startswith("Instruction"): out.append([])
        out[-1].append((t, round(ts, 2)))
    return out


n_steps = same = 0; n_items = same_items = 0; ex = []
for vid in sorted(set(sim) & set(real)):
    a, b = segs(sim[vid]), segs(real[vid])
    for i, sa in enumerate(a):
        if i >= len(b): break
        sb = [x for x in b[i] if x[1] <= hor[vid][i] + 1e-6]
        n_steps += 1; same += sa == sb
        if sa != sb and len(ex) < 5: ex.append((vid, i, sa[:3], sb[:3]))
        n_items += max(len(sa), len(sb)); same_items += sum(x == y for x, y in zip(sa, sb))
print(f"videos {len(set(sim) & set(real))} steps {n_steps} identical steps {same} ({same / max(n_steps, 1):.1%}) "
      f"items {same_items}/{n_items} ({same_items / max(n_items, 1):.1%})")
for e in ex: print("DIFF", e)
