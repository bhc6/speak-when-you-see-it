"""Collect runs that finished after the audit of 11-15 (audit_1114/RESULTS_AUTHORITATIVE.md is frozen) into data_addendum_1116.json:
pre-registered extra seeds of '+ notdone' (ABL_noCL_s2/s3), random-moment negatives (ECT_unm_s0/s1), ECT trained on black frames
(ECT_blind_s0/s1). For each run that has a completed QICD test evaluation (DONE marker) it reads the last cumulative block of the
official eval.log, recomputes P / R / F1 from the counts, computes the margin with analysis_code/frontier_canon.py, and reads the
931 time-matched pairs if present. Runs that are not finished are skipped and listed.
Run: python3 make_addendum.py   (then make_tables.py / make_supp.py pick up whatever is present)"""
import glob, json, os, sys
ROOT = "DATA_ROOT"; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{ROOT}/analysis_code"); import frontier_canon as F
NAMES = ["ABL_noCL_s2", "ABL_noCL_s3", "ECT_unm_s0", "ECT_unm_s1", "ECT_blind_s0", "ECT_blind_s1"]
acc = lambda L: sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in L) / max(1, len(L))
out = {"note": "Runs finished after the frozen table of audit_1114/ (pre-registered in the experiment registry). QICD test36, official eval, last cumulative block; margin = F1 - frontier_canon.fr('qicd', IC); pair accuracies from the 931 time-matched pairs. Written by make_addendum.py."}
missing = []
for n in NAMES:
    ev = [d for d in glob.glob(f"{ROOT}/*/results/jobs/{n}*test36*") if os.path.exists(f"{d}/DONE") and os.path.exists(f"{d}/eval.log")]
    if not ev: missing.append(n); continue
    d = sorted(ev)[0]; blk = open(f"{d}/eval.log").read().split("<<<")[-1]
    ic = float(blk.split("IC-Acc:")[1].split()[0]); tp, fp, fn = (int(blk.split(f"mistake_{k}:")[1].split(",")[0].split()[0]) for k in ("tp", "fp", "fn"))
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0; f1 = 2 * P * R / (P + R) if P + R else 0.0
    e = dict(source_eval=os.path.relpath(d, ROOT) + "/eval.log", qicd_ic=ic, qicd_tp_fp_fn=[tp, fp, fn], qicd_P=round(P, 3), qicd_R=round(R, 3), qicd_f1=round(f1, 3), qicd_margin=round(f1 - F.fr("qicd", ic), 3))
    ep = glob.glob(f"{ROOT}/*/results/jobs/epairs/{n}.json")
    if ep and os.path.exists(ep[0][:-5] + ".done"):
        x = json.load(open(ep[0])); e.update(source_epairs=os.path.relpath(ep[0], ROOT), n_pairs=len(x))
        for q, tag in (("instruction_end", "done"), ("mistake_inference", "mis")):
            L = [(p["p_pos"], p["p_neg"]) for p in x if p["q"] == q]; e[f"{tag}_pairacc"] = round(acc(L), 3)
            e[f"{tag}_both"] = round(sum(a > .5 and b < .5 for a, b in L) / max(1, len(L)), 3); e[f"{tag}_neg_yes"] = round(sum(b > .5 for a, b in L) / max(1, len(L)), 3)
    out[n] = e
json.dump(out, open(f"{HERE}/data_addendum_1116.json", "w"), indent=1)
for n in NAMES:
    if n in out: print(f"{n}: {out[n]['qicd_ic']} / {out[n]['qicd_f1']}  margin {out[n]['qicd_margin']:+.3f}  pairs {out[n].get('mis_pairacc', '--')}")
print("not finished:", missing or "none")
