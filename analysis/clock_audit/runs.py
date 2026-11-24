"""Frozen test predictions used by the 11-22 clock audit, with the IC-Acc / F1 of record (data/RESULTS_AUTHORITATIVE.md).
machineN/ are result folders of the evaluation machines under DATA_ROOT. Every run is re-scored with the local official
eval before use (audit.py stops on any mismatch)."""
B = "DATA_ROOT/"
J = lambda p: B + p + "/pred.json"
RUNS = {
    "qicd": [
        ("zero-shot 2B, official prompt", "zs", B + "results/test_vl_greedy.json", 4.2, 0.074),
        ("zero-shot 2B, memory prompt", "zs", B + "results/test_vl_mem.json", 10.5, 0.146),
        ("Qwen3.8-Max zero-shot", "zs", J("results/jobs/API_q38max_test36"), 23.9, 0.189),
        ("plain fine-tuning", "ft", B + "results/T1_test36.json", 52.1, 0.0),
        ("no neg., re-balanced", "ft", J("machine3/results/jobs/T1v2_2400_greedy_test36"), 27.7, 0.321),
        ("no paired neg. s0", "noneg", J("results/jobs/ABL_none_greedy_test36"), 23.1, 0.258),
        ("no paired neg. s1", "noneg", J("machine5/results/jobs/ABL_none_s1_greedy_test36"), 29.8, 0.254),
        ("+ notdone s0", "notdone", J("machine7/results/jobs/ABL_noCL_greedy_test36"), 31.1, 0.275),
        ("+ notdone s1", "notdone", J("machine7/results/jobs/ABL_noCL_s1_greedy_test36"), 32.8, 0.271),
        ("+ clean s0", "clean", J("results/jobs/ABL_noND_greedy_test36"), 43.3, 0.302),
        ("+ clean s1", "clean", J("machine3/results/jobs/ABL_noND_s1_greedy_test36"), 43.3, 0.27),
        ("ECT s0", "ect", J("results/final/sft_greedy_test36"), 41.2, 0.217),
        ("ECT s1", "ect", J("machine4/results/seeds/final/s1_greedy_test36"), 39.9, 0.284),
        ("ECT s2", "ect", J("machine5/results/seeds/final/s2_greedy_test36"), 39.5, 0.267),
        ("ECT s3", "ect", J("machine7/results/jobs/s3_greedy_test36"), 42.4, 0.221),
        ("random-moment neg. s0", "unm", J("machine5/results/jobs/ECT_unm_s0_greedy_test36"), 41.2, 0.292),
        ("random-moment neg. s1", "unm", J("machine4/results/jobs/ECT_unm_s1_greedy_test36"), 36.1, 0.271),
        ("ECT on black frames s0", "blind", J("machine7/results/jobs/ECT_blind_s0_greedy_test36"), 10.1, 0.0),
        ("ECT on black frames s1", "blind", J("machine3/results/jobs/ECT_blind_s1_greedy_test36"), 18.1, 0.0),
    ],
    "mc": [
        ("zero-shot 2B, official prompt", "zs", J("machine0/results/train/evals/R0_mc"), 14.0, 0.147),
        ("zero-shot 2B, memory prompt", "zs", J("machine0/results/final/r1_mem_mc"), 14.4, 0.215),
        ("Qwen3.8-Max zero-shot", "zs", J("machine11/results/jobs/API_q38max_mc"), 37.9, 0.324),
        ("plain fine-tuning", "ft", J("machine0/results/train/evals/T1_mc"), 60.2, 0.0),
        ("no neg., re-balanced", "ft", J("machine4/results/jobs/T1v2_2400_greedy_mc"), 47.7, 0.318),
        ("no paired neg. s0", "noneg", J("machine4/results/jobs/ABL_none_greedy_mc"), 47.7, 0.341),
        ("no paired neg. s1", "noneg", J("machine5/results/jobs/ABL_none_s1_greedy_mc"), 48.9, 0.333),
        ("+ notdone s0", "notdone", J("machine8/results/jobs/ABL_noCL_greedy_mc"), 56.8, 0.253),
        ("+ notdone s1", "notdone", J("machine7/results/jobs/ABL_noCL_s1_greedy_mc"), 51.1, 0.314),
        ("+ clean s0", "clean", J("machine0/results/jobs/ABL_noND_greedy_mc"), 54.5, 0.222),
        ("+ clean s1", "clean", J("machine3/results/jobs/ABL_noND_s1_greedy_mc"), 54.2, 0.222),
        ("ECT s0", "ect", J("machine0/results/final/sft_greedy_mc"), 67.0, 0.208),
        ("ECT s1", "ect", J("machine4/results/seeds/final/s1_greedy_mc"), 54.2, 0.228),
        ("ECT s2", "ect", J("machine5/results/seeds/final/s2_greedy_mc"), 59.1, 0.278),
        ("ECT s3", "ect", J("machine5/results/jobs/s3_greedy_mc"), 55.7, 0.271),
        ("random-moment neg. s0", "unm", J("machine8/results/jobs/ECT_unm_s0_greedy_mc"), 59.5, 0.272),
        ("random-moment neg. s1", "unm", J("machine8/results/jobs/ECT_unm_s1_greedy_mc"), 57.6, 0.261),
    ],
}
