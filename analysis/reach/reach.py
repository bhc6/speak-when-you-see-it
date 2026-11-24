"""How much of each main-table result comes from steps a run never reaches (it never confirms a step, so later steps are never
instructed)? Per run: recordings stopped early, steps reached, official IC-Acc, IC-Acc over reached mistake-free steps only, and the
share of mistake-free steps lost by not reaching them. Per-step logic = official eval.py (checked: totals equal the official run).
python3 reach.py -> out/reach.json"""
import sys, os, json, types
_t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)); sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); W = 15.0
J = lambda p: B + p + "/pred.json"
RUNS = {"qicd": {"Qwen3-VL-2B zero-shot": B + "results/test_vl_greedy.json", "Qwen3-VL-2B + memory prompt": B + "results/test_vl_mem.json",
                 "InternVL3.5-2B zero-shot": J("machine3/results/jobs/C1_ivl2b_zs_test36"), "LLaVA-OV-7B zero-shot": J("machine7/results/jobs/C5_llava7b_zs_test36"),
                 "Qwen2.5-VL-7B zero-shot": J("machine5/results/jobs/qicd_q25_mem_test36"), "Qwen3.5-397B": J("machine5/results/jobs/API_q35_397a17_test36"),
                 "Qwen3.8-Max": J("results/jobs/API_q38max_test36"),
                 "ECT s0": J("results/final/sft_greedy_test36"), "ECT s1": J("machine4/results/seeds/final/s1_greedy_test36"), "ECT s2": J("machine5/results/seeds/final/s2_greedy_test36"), "ECT s3": J("machine7/results/jobs/s3_greedy_test36"),
                 "no paired neg. s0": J("results/jobs/ABL_none_greedy_test36"), "no paired neg. s1": J("machine5/results/jobs/ABL_none_s1_greedy_test36"),
                 "ECT Qwen3-VL-4B": J("machine8/results/jobs/T1v4_4b_greedy_test36"), "ECT InternVL3.5-2B": J("machine4/results/jobs/C2_ivl2b_greedy_test36"), "ECT LLaVA-OV-7B": J("machine0/results/jobs/C6_llava7b_greedy_test36")},
        "mc": {"Qwen3-VL-2B zero-shot": J("machine0/results/train/evals/R0_mc"), "Qwen3-VL-2B + memory prompt": J("machine0/results/final/r1_mem_mc"),
               "InternVL3.5-2B zero-shot": J("machine3/results/jobs/C1_ivl2b_zs_mc"), "LLaVA-OV-7B zero-shot": J("machine11/results/jobs/C5_llava7b_zs_mc"),
               "Qwen2.5-VL-7B zero-shot": J("machine0/results/jobs/mc_q25_mem"), "Qwen3.5-397B": J("machine5/results/jobs/API_q35_397a17_mc"), "Qwen3.8-Max": J("machine11/results/jobs/API_q38max_mc"),
               "ECT s0": J("machine0/results/final/sft_greedy_mc"), "ECT s1": J("machine4/results/seeds/final/s1_greedy_mc"), "ECT s2": J("machine5/results/seeds/final/s2_greedy_mc"), "ECT s3": J("machine5/results/jobs/s3_greedy_mc"),
               "no paired neg. s0": J("machine4/results/jobs/ABL_none_greedy_mc"), "no paired neg. s1": J("machine5/results/jobs/ABL_none_s1_greedy_mc"),
               "ECT Qwen3-VL-4B": J("machine8/results/jobs/T1v4_4b_greedy_mc"), "ECT InternVL3.5-2B": J("machine8/results/jobs/C2_ivl2b_greedy_mc"), "ECT LLaVA-OV-7B": J("machine11/results/jobs/C6_llava7b_greedy_mc")}}
res = {}
for bench, runs in RUNS.items():
    ds = L.load("neurips2025" if bench == "qicd" else "ego_mc_bench", "test"); res[bench] = {}
    for name, p in runs.items():
        if not os.path.exists(p): print("MISSING", bench, name, p); continue
        preds = json.load(open(p)); off, _ = L.run(ds, preds); hit = den = den_r = steps = reached = stuck = 0
        for pr in preds:
            g = L.ev.get_ground_truth_data(ds, pr["video_id"]); segs, segt = L.ev.segment_pred_feedbacks(pr["pred_texts"], pr["pred_timestamps"])
            n = len(g["texts"]); ni = min(sum(x.startswith("Instruction") for x in pr["pred_texts"]), n)   # reached = instructed (the official
            steps += n; reached += ni; stuck += ni < n                                       # segmentation also drops a last step with no output)
            for k in range(n):
                if g["has_mistake"][k]: continue
                den += 1
                if k >= ni: continue
                den_r += 1
                if k >= len(segs): continue
                P = list(zip(segs[k][1:], segt[k][1:]))
                if L.ev.check_if_correct_instruction_given(g["texts"][k][0], segs[k][0].replace("Instruct:", "").strip()) and len(P) == 1 and "Success:" in P[0][0] and abs(P[0][1] - max(g["texts_timestamps"][k])) < W: hit += 1
        ic = round(100 * hit / den, 1); assert abs(ic - off["ic"]) < 0.06, (bench, name, ic, off["ic"])
        res[bench][name] = dict(recordings=len(preds), stopped_early=stuck, steps_reached=reached, steps=steps, ic_official=off["ic"],
                                ic_reached_only=round(100 * hit / max(den_r, 1), 1), mistake_free_not_reached=den - den_r, mistake_free=den, f1=off["f1"])
        print(bench, name, res[bench][name])
json.dump(res, open(f"{OUT}/reach.json", "w"), indent=1)
