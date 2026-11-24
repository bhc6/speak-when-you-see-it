"""Every decimal number in the paper text and tables must occur in a source of record (audit tables, experiment registry, technical
report, writing design). Prints the numbers that do not; those are either derived values (list them in DERIVED with the derivation)
or errors. Run: python3 num_check.py"""
import glob, json, os, re
ROOT = "DATA_ROOT"; HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [f"{ROOT}/audit_1114/RESULTS_AUTHORITATIVE.md", f"{ROOT}/audit_1114/QWEN_RESULTS.md", f"{ROOT}/EXPERIMENTS.md", f"{ROOT}/paper/technical_report.md",
       f"{HERE}/../PAPER_WRITING_DESIGN.md", f"{ROOT}/audit_1114/epair_summary.json", f"{ROOT}/audit_1114/decision_decomp.json", f"{ROOT}/machine0/results/blind_frontier.txt",
       f"{ROOT}/refcheck/published_frontier_egomc.json", f"{ROOT}/refcheck/J_guideme_blind.md", f"{ROOT}/refcheck/A_streaming_interventions.md", f"{HERE}/data_addendum_1116.json", f"{HERE}/supp/supp_numbers.json", f"{HERE}/supp/blindstrong_numbers.json", f"{HERE}/supp/notdone_numbers.json", f"{ROOT}/analysis_code/notdone_audit/FINDINGS_1120.md", f"{ROOT}/audit_1114/compute_1118/GPU15_AUDIT.md", f"{ROOT}/audit_1114/compute_1118/compute.json", f"{ROOT}/analysis_code/budget_clean/out/budget_all.json", f"{HERE}/supp/f1_null.json", f"{ROOT}/audit_1114/metrics.json", f"{ROOT}/audit_1114/gprobe_summary.json",
       f"{ROOT}/analysis_code/eq_results_1112.txt", f"{ROOT}/analysis_code/comist_audit/comist_audit.out", f"{ROOT}/audit_1114/coverage_1117/unm_stats.json", f"{ROOT}/analysis_code/guideme_blind/results/summary.json", f"{ROOT}/analysis_code/blind_plus/boot_qicd.log", f"{ROOT}/analysis_code/blind_plus/boot_mc.log",
       f"{ROOT}/machine3/results/analysis/paper_analysis.json", f"{HERE}/fig/hr/meta.json", f"{HERE}/fig/cand_mistake/meta.json", f"{ROOT}/machine0/results/blind_stream_neurips2025_test.json", f"{ROOT}/machine0/results/blind_stream_ego_mc_bench_test.json"]
SRC += [f"{HERE}/supp/random_moment_counts.json", f"{ROOT}/analysis_code/reach/out/reach.json",
        f"{ROOT}/analysis_code/egomc_unscored/out/unscored.json"]
pool = " ".join(open(p, encoding="utf-8").read() for p in SRC).replace("−", "-")
for _f in glob.glob(f"{ROOT}/analysis_code/opcurve/*.json"): pool += " " + open(_f).read()
nums = set(re.findall(r"\d+\.\d+", pool)); nums |= {f"{float(x):.{d}f}" for x in list(nums) if len(x.split(".")[1]) > 3 for d in (1, 2, 3)}; nums |= {n.rstrip("0").rstrip(".") for n in nums} | {f"{float(n):.3f}" for n in nums} | {f"{float(n):.2f}" for n in nums if len(n.split(".")[1]) <= 2}
DERIVED = {  # value: derivation
 "26.5": "mean QICD IC of the two no-negative seeds (23.1, 29.8)", "0.256": "also mean F1 (0.258, 0.254)", "48.3": "mean Ego-MC IC (47.7, 48.9)", "0.337": "mean (0.341, 0.333)",
 "31.9": "mean (31.1, 32.8)", "0.273": "mean (0.275, 0.271)", "54.0": "mean (56.8, 51.1)", "0.283": "mean (0.253, 0.314)", "0.286": "mean (0.302, 0.270)", "54.4": "mean (54.5, 54.2)",
 "0.65": "mean pair acc. (0.628, 0.665)", "0.72": "mean (0.659, 0.774) / InternVL 0.718", "0.74": "mean (0.727, 0.760)", "0.76": "mean of four ECT seeds", "0.67": "0.668 / 0.671 rounded",
 "0.66": "0.657 rounded", "0.59": "0.592 rounded", "0.75": "0.751 / 0.745 rounded", "0.77": "0.769 rounded", "0.90": "0.903 rounded", "0.92": "0.917 rounded", "0.63": "0.628 rounded",
 "0.87": "0.871 rounded", "0.34": "0.337 rounded", "0.45": "0.449 rounded", "0.69": "0.687 rounded", "0.10": "0.102 rounded / published F1", "0.5": "chance", "0.6": "0.64 rounded (false alarms per mistake-free step, ECT)", "0.43": "ECT precision on Ego-MC, mean of 0.51/0.349/0.49/0.384", "0.32": "Qwen3.8-Max precision 128/394 = 0.325", "0.33": "precision without negatives on Ego-MC, mean of 0.314 and 0.337", "0.68": "r with notdone alone (EXPERIMENTS 11-11 23:00)", "0.30": "same entry", "0.40": "same entry", "0.03": "-0.025 rounded (no-negative seed 0 margin)", "0.01": "+0.007 rounded", "0.21": "+0.208 / +0.214 rounded", "0.50": "chance",
 "0.06": "+0.062 rounded", "0.11": "+0.107 rounded", "0.20": "+0.197 rounded / published F1", "0.28": "+0.278 rounded", "0.91": "0.914 rounded", "1.00": "1.000", "0.00": "0.000",
 "0.13": "0.127 rounded", "0.18": "0.179 rounded / published", "0.05": "0.05", "0.07": "0.065 rounded", "4.1": "59.0 / 14.4", "3.9": "40.8 / 10.5",
 "0.009": "mean QICD margin without negatives (-0.025, +0.007)", "0.211": "mean (+0.208, +0.214)", "0.038": "mean (+0.035, +0.041)", "0.242": "mean (+0.253, +0.231) / thresholded F1",
 "0.167": "mean (+0.183, +0.151)", "0.193": "mean (+0.195, +0.191)", "0.085": "mean of four ECT margins", "0.237": "mean of four ECT margins (Ego-MC)",
 "34.0": "mean QICD IC of +notdone, 4 seeds (31.1, 32.8, 31.1, 40.8)", "0.297": "mean F1 (0.275, 0.271, 0.366, 0.278)", "0.080": "mean margin (0.035, 0.041, 0.126, 0.116)",
 "0.14": "+0.137 rounded (thresholded, QICD)", "0.17": "+0.167 rounded", "0.08": "+0.082 rounded", "0.24": "+0.242 rounded"}
bad = {}
for f in sorted(glob.glob(f"{HERE}/sec/*.tex") + glob.glob(f"{HERE}/tab/*.tex") + [f"{HERE}/supp.tex"] + glob.glob(f"{HERE}/supp/*.tex")):
    for ln, line in enumerate(open(f, encoding="utf-8"), 1):
        if line.lstrip().startswith("%"): continue
        for n in re.findall(r"(?<![\w.])\d+\.\d+", line.replace("{", " ").replace("}", " ").replace("\\pm", " ").replace("$", " ")):
            if n not in nums and n not in DERIVED: bad.setdefault(n, []).append(f"{os.path.basename(f)}:{ln}")
for n, w in sorted(bad.items()): print(n, w[:4])
print("unverified numbers:", len(bad))
