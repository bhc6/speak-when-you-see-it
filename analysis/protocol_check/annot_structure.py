"""How each benchmark annotates a step with mistakes: per-step event sequences (S success, Ma action-aligned mistake,
Mu action-unaligned mistake), whether a step with mistakes ever carries a success, and whether the same instruction is
issued again after it. Reads the ground-truth dumps of the official loaders. python3 annot_structure.py"""
import json, collections, sys
I = "DATA_ROOT/analysis_code/blind_strong/inputs/"
SETS = {"QICD Advanced Planning test": ("gt_dump_1118.json", "neurips2025_test"), "QICD Main test": ("gt_dump_main_1119.json", "neurips2025_test"),
        "Ego-MC-Bench test": ("gt_dump_1118.json", "ego_mc_bench_test")}
out = {}
for name, (f, k) in SETS.items():
    A = json.load(open(I + f))[k]; types = collections.Counter(); seq = collections.Counter(); follow = collections.Counter(); ex = None
    for a in A:
        segs = []
        for t, x, y in zip(a["output_timestamps"], a["output_texts"], a["output_types"]):
            types[y] += 1
            if y == "instruction": segs.append([x, [], t])
            elif "finish_all" not in y and segs: segs[-1][1].append((y, t, x))
        for i, (ins, ev, t0) in enumerate(segs):
            kinds = ["S" if "success" in y else ("Mu" if "unaligned" in y else ("Ma" if "aligned" in y else "M?")) for y, _, _ in ev]
            seq[" > ".join(kinds) or "(none)"] += 1
            if kinds and kinds[-1] != "S":
                rep = i + 1 < len(segs) and segs[i + 1][0] == ins; follow[(kinds[-1], "same instruction again" if rep else "next instruction")] += 1
                if rep and ex is None and i + 1 < len(segs) and segs[i + 1][1] and "success" in segs[i + 1][1][-1][0]:
                    ex = dict(video=a["video_id"], instruction=ins, first=[t0] + [(round(t, 1), y.split("_mistake")[0][-9:], x[:70]) for y, t, x in ev],
                              again=[segs[i + 1][2]] + [(round(t, 1), "success", x[:60]) for y, t, x in segs[i + 1][1]])
    mis_steps = sum(v for s, v in seq.items() if "M" in s); succ_after_mis = sum(v for s, v in seq.items() if "M" in s and "S" in s)
    out[name] = dict(videos=len(A), steps=sum(seq.values()), event_types=dict(types), sequences=dict(seq.most_common()), steps_with_mistakes=mis_steps,
                     mistake_steps_with_a_success=succ_after_mis, after_a_mistake_step=dict((f"{a} -> {b}", v) for (a, b), v in follow.items()), example=ex)
    print(f"== {name}: {len(A)} videos, {sum(seq.values())} steps, {mis_steps} with mistakes, of which {succ_after_mis} also carry a success")
    print("   sequences:", seq.most_common()); print("   after a step ending in a mistake:", dict(follow)); print("   example:", ex)
    print("   types:", {k.replace("feedback_", ""): v for k, v in types.items()})
json.dump(out, open("out/annot_structure.json", "w"), indent=1, ensure_ascii=False)
