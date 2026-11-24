"""What do the notdone negatives add? Step-level statistics from the existing greedy runs (post-hoc, exploratory; no new model run).
For every mistake-free step that the official evaluation scores (instruction matched), each run gives: whether it raises an
alarm, and the time of its confirmation relative to the annotated step end (offset = t_success - t_end; on time: |offset| < 15 s,
the official window; early: offset <= -15; late: offset >= 15 or never).
Groups (2 seeds each unless stated): ECT = clean + notdone (4 seeds), + clean, + notdone, no paired negatives.
Stratification: how long the step actually takes relative to what the clock predicts, ratio = annotated duration / median training
duration of the same instruction (QICD; Ego-MC instructions are unseen, so the global training median is used there).
Paired flips: steps on time for every ECT seed and early for every "+ clean" seed, and the reverse.
python notdone_stats.py OUT_DIR"""
import sys, os, json, statistics as st, collections, random
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
RUNS = {"qicd": {"ECT": ["results/final/sft_greedy_test36", "machine4/results/seeds/final/s1_greedy_test36", "machine5/results/seeds/final/s2_greedy_test36", "machine7/results/jobs/s3_greedy_test36"],
                 "+ clean": ["results/jobs/ABL_noND_greedy_test36", "machine3/results/jobs/ABL_noND_s1_greedy_test36"],
                 "+ notdone": ["machine7/results/jobs/ABL_noCL_greedy_test36", "machine7/results/jobs/ABL_noCL_s1_greedy_test36"],
                 "no neg.": ["results/jobs/ABL_none_greedy_test36", "machine5/results/jobs/ABL_none_s1_greedy_test36"]},
        "mc": {"ECT": ["machine0/results/final/sft_greedy_mc", "machine4/results/seeds/final/s1_greedy_mc", "machine5/results/seeds/final/s2_greedy_mc", "machine5/results/jobs/s3_greedy_mc"],
               "+ clean": ["machine0/results/jobs/ABL_noND_greedy_mc", "machine3/results/jobs/ABL_noND_s1_greedy_mc"],
               "+ notdone": ["machine8/results/jobs/ABL_noCL_greedy_mc", "machine7/results/jobs/ABL_noCL_s1_greedy_mc"],
               "no neg.": ["machine4/results/jobs/ABL_none_greedy_mc", "machine5/results/jobs/ABL_none_s1_greedy_mc"]}}
W = 15.0
def norm(s): return " ".join(s.strip().lower().rstrip(".").split())
tr = L.load("neurips2025", "train"); dur = collections.defaultdict(list)
for a in tr.annotations:
    g = L.ev.get_ground_truth_data(tr, a["video_id"])
    for tx, ts in zip(g["texts"], g["texts_timestamps"]): dur[norm(tx[0])].append(max(ts) - ts[0])
GMED = st.median([d for v in dur.values() for d in v]); MED = lambda s: st.median(dur[norm(s)]) if dur.get(norm(s)) else GMED

def steps_of(ds, pred):
    """per (video, step) of mistake-free scored steps: dict(start, end, dur, prior, alarm, offset)"""
    out = {}
    for item in pred:
        v = item["video_id"]; g = L.ev.get_ground_truth_data(ds, v)
        segs, segt = L.ev.segment_pred_feedbacks(item["pred_texts"], item["pred_timestamps"])
        for k in range(len(g["texts"])):
            if g["has_mistake"][k]: continue
            texts, ts = g["texts"][k], g["texts_timestamps"][k]
            if k >= len(segs) or not L.ev.check_if_correct_instruction_given(texts[0], segs[k][0].replace("Instruct:", "").strip()): continue
            P = list(zip(segs[k][1:], segt[k][1:])); fb = any("Feedback" in x and "all the steps" not in x for x, _ in P)
            sc = [t for x, t in P if "Success:" in x]; st_, en = ts[0], max(ts)
            out[(v, k)] = dict(start=st_, end=en, dur=en - st_, prior=MED(texts[0]), instr=texts[0], alarm=fb,
                               offset=(sc[0] - en) if sc else None, n_out=len(P))
    return out
def cls(o):
    if o["alarm"]: return "alarm"
    if o["offset"] is None: return "never"
    if o["n_out"] == 1 and abs(o["offset"]) < W: return "on time"
    return "early" if o["offset"] <= -W else ("late" if o["offset"] >= W else "other")
res = {}
for ds_name, groups in RUNS.items():
    ds = L.load("neurips2025" if ds_name == "qicd" else "ego_mc_bench", "test")
    S = {g: [steps_of(ds, json.load(open(B + p + "/pred.json"))) for p in ps] for g, ps in groups.items()}
    keys = sorted(set.intersection(*[set(s) for runs in S.values() for s in runs]))
    ratio = {k: S["ECT"][0][k]["dur"] / max(S["ECT"][0][k]["prior"], 1) for k in keys}
    q = sorted(ratio.values()); t1, t2 = q[len(q) // 3], q[2 * len(q) // 3]
    stratum = {k: ("shorter than typical" if ratio[k] < t1 else "typical" if ratio[k] < t2 else "longer than typical") for k in keys}
    R = {"n_steps": len(keys), "tercile_cuts": [t1, t2], "groups": {}}
    for g, runs in S.items():
        rows = {}
        for strat in ("all", "shorter than typical", "typical", "longer than typical"):
            K = [k for k in keys if strat == "all" or stratum[k] == strat]; c = collections.Counter(cls(r[k]) for r in runs for k in K); n = len(K) * len(runs)
            offs = [r[k]["offset"] for r in runs for k in K if not r[k]["alarm"] and r[k]["offset"] is not None]
            rows[strat] = dict(n=len(K), **{x: round(100 * c[x] / n, 1) for x in ("on time", "early", "late", "never", "alarm", "other")},
                               median_offset=round(st.median(offs), 1) if offs else None)
        # does the confirmation follow the actual duration beyond the prior? partial correlation of t_confirm-start with duration given prior
        xs = [(r[k]["offset"] + r[k]["dur"], r[k]["dur"], r[k]["prior"]) for r in runs for k in keys if not r[k]["alarm"] and r[k]["offset"] is not None]
        def corr(a, b):
            ma, mb = st.mean(a), st.mean(b); sa = sum((x - ma) ** 2 for x in a) ** .5; sb = sum((y - mb) ** 2 for y in b) ** .5
            return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / max(sa * sb, 1e-9)
        c_t_d, c_t_p, c_d_p = corr([x[0] for x in xs], [x[1] for x in xs]), corr([x[0] for x in xs], [x[2] for x in xs]), corr([x[1] for x in xs], [x[2] for x in xs])
        partial = (c_t_d - c_t_p * c_d_p) / max(((1 - c_t_p ** 2) * (1 - c_d_p ** 2)) ** .5, 1e-9)
        rows["corr_confirm_vs_duration"] = round(c_t_d, 3); rows["corr_confirm_vs_prior"] = round(c_t_p, 3); rows["partial_corr_duration_given_prior"] = round(partial, 3)
        R["groups"][g] = rows
    # paired flips ECT vs + clean
    def all_cls(g, k): return [cls(r[k]) for r in S[g]]
    fixed = [k for k in keys if all(c == "on time" for c in all_cls("ECT", k)) and all(c == "early" for c in all_cls("+ clean", k))]
    broken = [k for k in keys if all(c == "early" for c in all_cls("ECT", k)) and all(c == "on time" for c in all_cls("+ clean", k))]
    R["flips"] = dict(ect_on_time_clean_early=len(fixed), ect_early_clean_on_time=len(broken),
                      fixed=[dict(video=k[0], step=k[1], dur=round(S["ECT"][0][k]["dur"], 1), prior=round(S["ECT"][0][k]["prior"], 1), instr=S["ECT"][0][k]["instr"],
                                  clean_offsets=[round(r[k]["offset"], 1) for r in S["+ clean"]], ect_offsets=[round(r[k]["offset"], 1) for r in S["ECT"]]) for k in fixed],
                      broken=[dict(video=k[0], step=k[1], dur=round(S["ECT"][0][k]["dur"], 1), instr=S["ECT"][0][k]["instr"]) for k in broken],
                      fixed_by_stratum=dict(collections.Counter(stratum[k] for k in fixed)), broken_by_stratum=dict(collections.Counter(stratum[k] for k in broken)))
    res[ds_name] = R
    print(f"\n=== {ds_name}: {len(keys)} mistake-free steps scored in every run; ratio terciles {t1:.2f} / {t2:.2f}")
    for g, rows in R["groups"].items():
        print(f"{g:10s} " + " | ".join(f"{s[:7]}: on {rows[s]['on time']:4.1f} early {rows[s]['early']:4.1f} late {rows[s]['late']:4.1f} alarm {rows[s]['alarm']:4.1f}" for s in ("all", "shorter than typical", "typical", "longer than typical")))
        print(f"{'':10s} corr(confirm, duration) {rows['corr_confirm_vs_duration']}  corr(confirm, prior) {rows['corr_confirm_vs_prior']}  partial(duration | prior) {rows['partial_corr_duration_given_prior']}")
    print("flips: ECT on time & +clean early:", len(fixed), dict(collections.Counter(stratum[k] for k in fixed)), "| reverse:", len(broken), dict(collections.Counter(stratum[k] for k in broken)))
json.dump(res, open(f"{OUT}/notdone_stats.json", "w"), indent=1, default=str)
