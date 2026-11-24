"""Ego-MC steps annotated as mistake(s) then success: confirmation timing where the benchmark does not score it (see PLAN.md).
python3 unscored.py -> out/unscored.json"""
import sys, os, json, types, random, collections, statistics as st
_t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)); sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); W = 15.0
RUNS = {"ECT": ["machine0/results/final/sft_greedy_mc", "machine4/results/seeds/final/s1_greedy_mc", "machine5/results/seeds/final/s2_greedy_mc", "machine5/results/jobs/s3_greedy_mc"],
        "+ clean": ["machine0/results/jobs/ABL_noND_greedy_mc", "machine3/results/jobs/ABL_noND_s1_greedy_mc"],
        "+ notdone": ["machine8/results/jobs/ABL_noCL_greedy_mc", "machine7/results/jobs/ABL_noCL_s1_greedy_mc"],
        "no neg.": ["machine4/results/jobs/ABL_none_greedy_mc", "machine5/results/jobs/ABL_none_s1_greedy_mc"],
        "zero-shot (memory prompt)": ["machine0/results/final/r1_mem_mc"]}
ds = L.load("ego_mc_bench", "test"); GT = {}
for a in ds.annotations:
    g = L.ev.get_ground_truth_data(ds, a["video_id"])
    for k in range(len(g["texts"])):
        ty, ts = g["texts_types"][k], g["texts_timestamps"][k]
        mis = [t for t, y in zip(ts, ty) if "mistake" in y]; suc = [t for t, y in zip(ts, ty) if "success" in y]
        GT[(a["video_id"], k)] = dict(instr=g["texts"][k][0], start=ts[0], mistakes=mis, success=suc, has_mistake=bool(g["has_mistake"][k]),
                                      target=bool(g["has_mistake"][k]) and bool(suc) and max(suc) > max(mis))
POP = sorted(k for k, v in GT.items() if v["target"]); FREE = sorted(k for k, v in GT.items() if not v["has_mistake"] and v["success"])
def outcomes(path):
    """per (video, k) reached with matched instruction: category, before-last-mistake flag, alarms"""
    res = {}
    for item in json.load(open(B + path + "/pred.json")):
        v = item["video_id"]; segs, segt = L.ev.segment_pred_feedbacks(item["pred_texts"], item["pred_timestamps"])
        for k in range(len(segs)):
            s = GT.get((v, k))
            if s is None or not L.ev.check_if_correct_instruction_given(s["instr"], segs[k][0].replace("Instruct:", "").strip()): continue
            P = list(zip(segs[k][1:], segt[k][1:])); c = next((t for x, t in P if "Success:" in x), None)
            alarms = [t for x, t in P if "Feedback" in x and "all the steps" not in x]
            T = max(s["success"]) if s["success"] else None
            cat = None if T is None else ("never" if c is None else ("on_time" if abs(c - T) < W else ("early" if c <= T - W else "late")))
            res[(v, k)] = dict(cat=cat, before_last=c is not None and bool(s["mistakes"]) and c < max(s["mistakes"]), alarms=alarms)
    return res
O = {g: [outcomes(p) for p in ps] for g, ps in RUNS.items()}
def rate(g, keys, f):
    vals = []
    for run in O[g]:
        ks = [k for k in keys if k in run]; vals.append(100 * sum(f(run[k]) for k in ks) / max(len(ks), 1))
    return sum(vals) / len(vals)
F = {"on_time": lambda r: r["cat"] == "on_time", "early": lambda r: r["cat"] == "early", "late_or_never": lambda r: r["cat"] in ("late", "never"),
     "before_last_mistake": lambda r: r["before_last"]}
res = dict(population=len(POP), videos=len({v for v, _ in POP}), reached={g: [len([k for k in POP if k in run]) for run in O[g]] for g in O}, levels={}, free_steps_levels={})
for g in O:
    res["levels"][g] = {m: round(rate(g, POP, f), 1) for m, f in F.items() if m != "x"}
    res["free_steps_levels"][g] = {m: round(rate(g, FREE, f), 1) for m, f in F.items() if m != "before_last_mistake"}
# 1. ECT vs + clean, bootstrap over videos
vids = sorted({v for v, _ in POP}); byv = collections.defaultdict(list)
for k in POP: byv[k[0]].append(k)
def diff(sel, m, a="ECT", b="+ clean"):
    keys = [k for v in sel for k in byv[v]]; return rate(a, keys, F[m]) - rate(b, keys, F[m])
res["ect_minus_clean"] = {}
for m in ("on_time", "early", "late_or_never", "before_last_mistake"):
    rng = random.Random(0); bs = sorted(diff([rng.choice(vids) for _ in vids], m) for _ in range(2000))
    res["ect_minus_clean"][m] = dict(diff=round(diff(vids, m), 1), ci95=[round(bs[50], 1), round(bs[1949], 1)])
# 2. best fixed-elapsed blind rule, tuned on this population (optimistic)
def blind(E, keys):
    hit = early = 0
    for k in keys:
        s = GT[k]; i0 = int(s["start"] * 2); j = 1
        while (i0 + 10 * j) / 2 - s["start"] < E: j += 1
        c = (i0 + 10 * j) / 2; T = max(s["success"]); hit += abs(c - T) < W; early += c <= T - W
    return 100 * hit / len(keys), 100 * early / len(keys)
grid = {E: blind(E, POP) for E in range(5, 305, 5)}; Eb = max(grid, key=lambda e: grid[e][0])
res["best_blind_fixed_elapsed"] = dict(E=Eb, on_time=round(grid[Eb][0], 1), early=round(grid[Eb][1], 1), note="E tuned on this population (optimistic for the rule); all population steps")
# 3. tertiles of annotated duration
durs = sorted(max(GT[k]["success"]) - GT[k]["start"] for k in POP); q1, q2 = durs[len(durs) // 3], durs[2 * len(durs) // 3]
tert = {"short": [k for k in POP if max(GT[k]["success"]) - GT[k]["start"] < q1], "middle": [k for k in POP if q1 <= max(GT[k]["success"]) - GT[k]["start"] < q2],
        "long": [k for k in POP if max(GT[k]["success"]) - GT[k]["start"] >= q2]}
res["tertiles"] = {t: dict(n=len(ks), dur_range=[round(min(max(GT[k]["success"]) - GT[k]["start"] for k in ks), 1), round(max(max(GT[k]["success"]) - GT[k]["start"] for k in ks), 1)],
                           blind_best_overall_E=round(blind(Eb, ks)[0], 1),
                           **{g: {m: round(rate(g, ks, F[m]), 1) for m in ("on_time", "early", "late_or_never")} for g in ("ECT", "+ clean", "+ notdone", "no neg.")}) for t, ks in tert.items()}
# secondary: non-first mistakes caught in steps with >= 2 mistakes
multi = [k for k in POP if len(GT[k]["mistakes"]) >= 2]
def caught(g):
    vals = []
    for run in O[g]:
        n = h = 0
        for k in multi:
            if k not in run: continue
            M = GT[k]["mistakes"]; reg = set()
            for t in run[k]["alarms"]:
                j = min(range(len(M)), key=lambda i: abs(M[i] - t))
                if abs(M[j] - t) < W: reg.add(j)
            first = min(range(len(M)), key=lambda i: M[i]); n += len(M) - 1; h += len([j for j in reg if j != first])
        vals.append(100 * h / max(n, 1))
    return round(sum(vals) / len(vals), 1)
res["secondary_nonfirst_mistakes_caught"] = dict(steps=len(multi), **{g: caught(g) for g in O})
json.dump(res, open(f"{OUT}/unscored.json", "w"), indent=1); print(json.dumps(res, indent=1))
# added after the first run (same population and outcome): interval for ECT and the other groups minus the best blind rule (E fixed at Eb)
def blind_hits(keys): return {k: blind(Eb, [k])[0] / 100 for k in keys}
BH = blind_hits(POP); res["minus_best_blind"] = {}
for g in O:
    def d(sel): keys = [k for v in sel for k in byv[v]]; return rate(g, keys, F["on_time"]) - 100 * sum(BH[k] for k in keys) / len(keys)
    rng = random.Random(0); bs = sorted(d([rng.choice(vids) for _ in vids]) for _ in range(2000))
    res["minus_best_blind"][g] = dict(diff=round(d(vids), 1), ci95=[round(bs[50], 1), round(bs[1949], 1)])
print("minus best blind:", res["minus_best_blind"]); json.dump(res, open(f"{OUT}/unscored.json", "w"), indent=1)
# added 11-21 after the pre-registered part (descriptive, single runs): other backbones and training variants; a text-aware blind rule
import re
EXTRA = {"ECT InternVL3.5-2B": "machine8/results/jobs/C2_ivl2b_greedy_mc", "InternVL3.5-2B no paired neg.": "machine4/results/jobs/C3_ivl2b_none_greedy_mc",
         "InternVL3.5-2B zero-shot": "machine3/results/jobs/C1_ivl2b_zs_mc", "ECT LLaVA-OV-7B": "machine11/results/jobs/C6_llava7b_greedy_mc", "ECT Qwen3-VL-4B": "machine8/results/jobs/T1v4_4b_greedy_mc",
         "no neg., reweighted (T1v2)": "machine4/results/jobs/T1v2_2400_greedy_mc", "clean x0.75 (B1)": "machine7/results/jobs/B1_clean075_greedy_mc", "clean x3 (B2)": "machine8/results/jobs/B2_clean300_greedy_mc",
         "Qwen3.8-Max": "machine11/results/jobs/API_q38max_mc", "Qwen3.5-397B": "machine5/results/jobs/API_q35_397a17_mc"}
res["extra_single_runs"] = {}
for name, p in EXTRA.items():
    r = outcomes(p); ks = [k for k in POP if k in r]; c = collections.Counter(r[k]["cat"] for k in ks)
    res["extra_single_runs"][name] = dict(reached=len(ks), on_time=round(100 * c["on_time"] / len(ks), 1), early=round(100 * c["early"] / len(ks), 1),
                                          late_or_never=round(100 * (c["late"] + c["never"]) / len(ks), 1))
def stated(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(?:\d+\s*)?(second|sec|minute|min)s?\b", s.lower())
    return None if not m else float(m.group(1)) * (60 if m.group(2).startswith("min") else 1)
best = (0, None)
for E in range(5, 305, 5):
    for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        hit = 0
        for k in POP:
            s = GT[k]; d = stated(s["instr"]); target = E if d is None else a * d; i0 = int(s["start"] * 2); j = 1
            while (i0 + 10 * j) / 2 - s["start"] < target: j += 1
            hit += abs((i0 + 10 * j) / 2 - max(s["success"])) < W
        best = max(best, (100 * hit / len(POP), (E, a)))
res["text_aware_blind"] = dict(on_time=round(best[0], 1), E=best[1][0], factor=best[1][1], steps_with_stated_duration=sum(stated(GT[k]["instr"]) is not None for k in POP),
                               note="instructions stating a duration: confirm at factor x that duration; others at E; both tuned on the population")
json.dump(res, open(f"{OUT}/unscored.json", "w"), indent=1); print("extra:", res["extra_single_runs"]); print("text-aware blind:", res["text_aware_blind"])
