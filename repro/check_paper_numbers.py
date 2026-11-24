"""Recompute the headline numbers of the paper and the supplement from the bundled data, with the Python standard library only.
From the code/ directory:  python3 repro/check_paper_numbers.py   -> prints each check and ALL MATCH on success.
1. Every margin in data/RESULTS_AUTHORITATIVE.md equals F1 minus the rule frontier at that IC-Acc (analysis/frontier_canon.py);
   the record keeps rule margins, as the supplement tables do.
2. The ECT rows of Table 1: mean +- s.d. over four seeds and the range of margins against the paper's frontier (rules and the
   learned blind policy on the same inputs, analysis/frontier_v2.py), on both benchmarks.
3. The numbers of the post-hoc clock analyses quoted in supplement B, F and H (data/clock_audit/)."""
import json, os, re, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis")); import frontier_canon as F; import frontier_v2 as F2
fails = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol + 1e-9) if isinstance(want, (int, float)) else got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (paper: {want})")
    if not ok: fails.append(name)


rows = []
for l in open(os.path.join(ROOT, "data", "RESULTS_AUTHORITATIVE.md"), encoding="utf-8"):
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) == 8 and c[0] in ("qicd", "mc") and re.match(r"^[+-]\d", c[7]):
        rows.append((c[0], c[1], float(c[2]), float(c[5]), float(c[7])))
bad = [r for r in rows if abs(r[3] - F.fr(r[0], r[2]) - r[4]) > 0.0015]
check(f"1. rule margins recomputed from the rule frontier ({len(rows)} runs), mismatches", len(bad), 0)
for r in bad[:5]: print("     ", r, round(r[3] - F.fr(r[0], r[2]), 3))
R = {(r[0], r[1]): r for r in rows}
SEEDS = {"qicd": ["results/final/sft_greedy_test36", "machine4/results/seeds/final/s1_greedy_test36", "machine5/results/seeds/final/s2_greedy_test36", "machine7/results/jobs/s3_greedy_test36"],
         "mc": ["machine0/results/final/sft_greedy_mc", "machine4/results/seeds/final/s1_greedy_mc", "machine5/results/seeds/final/s2_greedy_mc", "machine5/results/jobs/s3_greedy_mc"]}
WANT = {"qicd": ("40.8", "1.3", "0.247", "0.033", "+0.01", "+0.06"), "mc": ("59.0", "5.7", "0.246", "0.034", "+0.17", "+0.28")}
for ds, runs in SEEDS.items():
    S = [R[(ds, r)] for r in runs]; ic = [s[2] for s in S]; f1 = [s[3] for s in S]; m = [F2.margin(ds, s[2], s[3]) for s in S]
    got = (f"{st.mean(ic):.1f}", f"{st.stdev(ic):.1f}", f"{st.mean(f1):.3f}", f"{st.stdev(f1):.3f}", f"{min(m):+.2f}", f"{max(m):+.2f}")
    check(f"2. Table 1, ECT on {ds}: IC mean, s.d., F1 mean, s.d., margin range", got, WANT[ds])
C = os.path.join(ROOT, "data", "clock_audit"); J = lambda n: json.load(open(os.path.join(C, n + ".json")))
pa = J("prior_auroc")
check("3. B: completion AUROC of raw elapsed time, QICD test", pa["neurips2025_test"]["completion"]["elapsed"], 0.525)
check("3. B: completion AUROC of elapsed time / training duration, QICD test", pa["neurips2025_test"]["completion"]["elapsed_over_prior"], 0.61, 0.005)
check("3. B: the same, QICD validation", pa["neurips2025_validation"]["completion"]["elapsed_over_prior"], 0.69, 0.005)
rp = J("replay")
check("3. F: replay of ECT s0, IC-Acc", rp["validation_ECT_s0_frozen"]["ic"], 42.0)
check("3. F: replay without paired negatives, IC-Acc", rp["validation_noneg_s0_frozen"]["ic"], 31.1)
check("3. F: per-check alarm rate, ECT / without paired negatives", (round(rp["mechanism|ECT s0"]["per_check_alarm_rate"], 3), round(rp["mechanism|no paired neg. s0"]["per_check_alarm_rate"], 3)), (0.073, 0.183))
check("3. F: voided steps, ECT / without paired negatives", (rp["mechanism|ECT s0"]["voided_actual"], rp["mechanism|no paired neg. s0"]["voided_actual"]), (0.265, 0.408))
h = [r for r in rp["hybrid|completion noneg + mistake ECT"] if r["tau_mis"] == 0.35][0]
check("3. F: completion of the variant with mistake probabilities of ECT, IC-Acc", h["ic"], 40.8)
best = {k: max(r["ic"] for r in rp[f"sweep|{k}"] if r["tau_mis"] > 1) for k in ("ECT s0", "no paired neg. s0")}
check("3. F: best IC-Acc without alarms, ECT / variant", (best["ECT s0"], best["no paired neg. s0"]), (61.3, 54.6))
pp = J("pair_prior")
check("3. E: time-matched pairs, blind policy with the instruction's typical duration (completion, mistake)", (pp["completion"]["elapsed_over_prior"], pp["mistake"]["elapsed_over_prior"]), (0.651, 0.539))
check("3. F: replay ending recordings at a step that outlasts the log, IC-Acc (ECT, variant)", (rp["no_continuation_ECT_s0"]["ic"], rp["no_continuation_noneg_s0"]["ic"]), (34.0, 25.2))
check("3. F: exchange at thresholds of 0.5, IC-Acc (variant, variant with ECT's mistake probabilities, ECT)", (rp["near_greedy|no paired neg."]["ic"], rp["near_greedy|completion noneg + mistake ECT"]["ic"], rp["near_greedy|ECT"]["ic"]), (23.1, 38.7, 39.1))
tr = J("train_pair_prior"); acc = lambda kind: sorted(v["within_pair_accuracy"] for k, v in tr.items() if k.startswith(kind + "|"))
check("3. H: training pairs separated by the duration prior, completion range", (round(acc("success")[0] * 100), round(acc("success")[-1] * 100)), (80, 81))
check("3. H: the same, mistake range", (round(acc("mistake")[0] * 100), round(acc("mistake")[-1] * 100)), (58, 59))
tc = J("tp_contrast")["ect_minus_noneg_mistake_auroc"]
check("3. H: ECT minus variant, mistake AUROC with original stamps", tc["ts1"][0], -0.130)
tp = J("tp_analyze")["ECT s0"]
check("3. H: completion AUROC of ECT with original / doubled / halved stamps", tuple(tp[f"auc_done_{c}"][0] for c in ("ts1", "ts2", "ts05")), (0.945, 0.942, 0.943))
print("ALL MATCH" if not fails else f"{len(fails)} MISMATCH(ES): {fails}")
sys.exit(1 if fails else 0)
