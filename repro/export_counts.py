"""Per-video counts behind the same-validation-alarm-cap table and the long-step table, exported from the
prediction files so that both tables can be recomputed without the predictions (reproduce_tables.py).
python export_counts.py OUT_DIR  -> budget_per_video.csv, long_steps_per_video.csv"""
import sys, os, json, csv
D = "DATA_ROOT/analysis_code"; DEST = os.path.abspath(sys.argv[1]); os.makedirs(DEST, exist_ok=True)
# ---- same validation alarm cap: thresholded ECT seed 0 and each variant replayed at its threshold (budget_clean.py / budget_all.py)
sys.argv = ["x", "TMP_PATH"]; exec(open(f"{D}/budget_clean/budget_clean.py").read().split("res = {}; runs = {}")[0])
bg = json.load(open(f"{D}/budget_clean/out/budget_all.json"))
VAR = {"No paired neg.": ("results/jobs/ABL_none_prob_test36", "machine4/results/jobs/ABL_none_prob_mc"),
       "+ notdone": ("machine7/results/jobs/ABL_noCL_prob_test36", "machine8/results/jobs/ABL_noCL_prob_mc"),
       "+ clean": ("results/jobs/ABL_noND_prob_test36", "machine3/results/jobs/ABL_noND_prob_mc")}
def per_video(ds, P):
    P = {p["video_id"]: p for p in P}; vids = sorted(P); out = {}
    cl = {v: sum(not x for x in L.ev.get_ground_truth_data(ds, v)["has_mistake"]) for v in vids}
    v0 = next(v for v in vids if cl[v]); r0, _ = L.run(ds, [P[v0]])
    for v in vids:
        if cl[v]: r, _ = L.run(ds, [P[v]]); out[v] = (round(r["ic"] * cl[v] / 100), cl[v], r["tp"], r["fp"], r["fn"])
        else: r, _ = L.run(ds, [P[v0], P[v]]); out[v] = (0, 0, r["tp"] - r0["tp"], r["fp"] - r0["fp"], r["fn"] - r0["fn"])
    return out
with open(f"{DEST}/budget_per_video.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["benchmark", "run", "tau", "video", "ic_hits", "mistake_free_steps", "tp", "fp", "fn"])
    for k, dv in (("qicd", "neurips2025"), ("mc", "ego_mc_bench")):
        ds = L.load(dv, "test")
        for v, c in per_video(ds, json.load(open(ECT[k] + "/pred.json"))).items(): w.writerow([k, "ECT (thresholded, seed 0)", "0.35", v, *c])
        for name, (pt, pm) in VAR.items():
            tau = next(r for r in bg["rows"] if r["variant"] == name and r["rule"] == "entries")["tau"]
            for v, c in per_video(ds, replay(load_run(B + (pt if k == "qicd" else pm)), tau)).items(): w.writerow([k, name, f"{tau:.2f}", v, *c])
# ---- long steps: long mistake-free steps, outcome counts per video and run (notdone_stats.py classes)
sys.argv = ["x", "TMP_PATH"]; exec(open(f"{D}/notdone_audit/notdone_stats.py").read().split("res = {}")[0])
with open(f"{DEST}/long_steps_per_video.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["benchmark", "group", "run", "video", "long_steps", "on_time", "early", "late", "never", "alarm", "other"])
    for dsn in ("mc", "qicd"):
        ds = L.load("ego_mc_bench" if dsn == "mc" else "neurips2025", "test"); S = {g: [steps_of(ds, json.load(open(B + p + "/pred.json"))) for p in ps] for g, ps in RUNS[dsn].items()}
        keys = sorted(set.intersection(*[set(s) for runs in S.values() for s in runs])); E0 = S["ECT"][0]; q = sorted(E0[k]["dur"] for k in keys); cut = q[2 * len(q) // 3]
        LONG = [k for k in keys if E0[k]["dur"] >= cut]
        for g, runs in S.items():
            for p, r in zip(RUNS[dsn][g], runs):
                for v in sorted({k[0] for k in LONG}):
                    K = [k for k in LONG if k[0] == v]; c = [cls(r[k]) for k in K]
                    w.writerow([dsn, g, p.split("/")[-1], v, len(K)] + [c.count(x) for x in ("on time", "early", "late", "never", "alarm", "other")])
print("written", DEST)
