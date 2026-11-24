"""Where do runs that stop early stop? In the per-step protocol only a confirmation issues the next instruction, so a
recording whose last issued instruction is never confirmed stops there. For each such recording we classify that step by
its annotation: mistake-free; ends with a mistake (the last event is a mistake and no success follows, the step a model
should arguably never confirm); or mistake then success. Frozen test predictions only.
Also per test set: steps that end with a mistake, as a share of all steps.
python3 stall_steps.py -> out/stall_steps.json"""
import json, os, collections
B = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(B + "analysis_code/blind_strong/inputs/gt_dump_1118.json"))
J = lambda p: B + p + "/pred.json"
RUNS = {"neurips2025_test": {"Qwen3-VL-2B + memory prompt": B + "results/test_vl_mem.json", "Qwen3.5-397B": J("machine5/results/jobs/API_q35_397a17_test36"),
                             "Qwen2.5-VL-7B": J("machine5/results/jobs/qicd_q25_mem_test36"),
                             **{f"ECT s{i}": J(p) for i, p in enumerate(("results/final/sft_greedy_test36", "machine4/results/seeds/final/s1_greedy_test36",
                                                                         "machine5/results/seeds/final/s2_greedy_test36", "machine7/results/jobs/s3_greedy_test36"))}},
        "ego_mc_bench_test": {"Qwen3-VL-2B + memory prompt": J("machine0/results/final/r1_mem_mc"), "Qwen3.5-397B": J("machine5/results/jobs/API_q35_397a17_mc"),
                              "Qwen2.5-VL-7B": J("machine0/results/jobs/mc_q25_mem"),
                              **{f"ECT s{i}": J(p) for i, p in enumerate(("machine0/results/final/sft_greedy_mc", "machine4/results/seeds/final/s1_greedy_mc",
                                                                          "machine5/results/seeds/final/s2_greedy_mc", "machine5/results/jobs/s3_greedy_mc"))}}}


def kind(types):
    ev = [t for t in types[1:]]
    if not any("mistake" in t for t in ev): return "mistake-free"
    last_m = max(i for i, t in enumerate(ev) if "mistake" in t)
    return "mistake then success" if any("success" in t for t in ev[last_m + 1:]) else "ends with a mistake"


out = {}
for k, runs in RUNS.items():
    GT = {g["video_id"]: g for g in G[k]}
    allk = collections.Counter(kind(ty) for g in G[k] for ty in g["instruction_segment_texts_types"])
    out[k] = {"steps": sum(allk.values()), "step_kinds": dict(allk), "runs": {}}
    for name, path in runs.items():
        c = collections.Counter(); stopped = 0
        for p in json.load(open(path)):
            g = GT[p["video_id"]]; n = sum(t.startswith("Instruction") for t in p["pred_texts"]); steps = g["instruction_segment_texts_types"]
            if n < len(steps):   # the run never reached the last step: it stopped at step n-1
                stopped += 1; c[kind(steps[n - 1])] += 1
        out[k]["runs"][name] = dict(stopped_recordings=stopped, recordings=len(GT), stopped_at=dict(c))
        print(k, name, stopped, dict(c))
    print(k, "all steps", dict(allk))
json.dump(out, open(HERE + "/out/stall_steps.json", "w"), indent=1)
