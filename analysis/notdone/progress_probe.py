"""Within-step progress sensitivity from the probability logs of thresholded runs (post-hoc, exploratory).
For each check of a mistake-free step before the model confirms it: P(yes) to the completion question and the progress of the step
(elapsed / annotated duration). Statistics per run:
  p_early : mean P(done) at checks with progress < 0.5 (the step is less than half done; a clock-only model is tempted here
            when the step runs long)
  p_late  : mean P(done) at checks with progress in [0.85, 1.0] (just before the annotated end)
  auroc   : P(done) separating checks at progress >= 0.85 from checks at progress < 0.5 (within the same steps)
Logs: plog{k}.jsonl per shard when present; otherwise the shared plog.jsonl of two concurrent shards is demultiplexed by replaying
each shard's known step sequence, and accepted only if the demultiplexed decisions reproduce every Success and Feedback of the
shard's predictions exactly.
python progress_probe.py OUT_DIR"""
import sys, os, json, glob, random, statistics as st
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
RUNS = {"qicd": {"ECT": ["results/final/sft_prob_test36", "machine4/results/seeds/final/s1_prob_test36", "machine5/results/seeds/final/s2_prob_test36", "machine6/results/seeds/final/s3_prob_test36"],
                 "+ clean": ["results/jobs/ABL_noND_prob_test36"], "+ notdone": ["machine7/results/jobs/ABL_noCL_prob_test36"], "no neg.": ["results/jobs/ABL_none_prob_test36"]},
        "mc": {"ECT": ["machine0/results/final/sft_prob_mc", "machine4/results/seeds/final/s1_prob_mc", "machine5/results/seeds/final/s2_prob_mc", "machine5/results/jobs/s3_prob_mc"],
               "+ clean": ["machine3/results/jobs/ABL_noND_prob_mc"], "+ notdone": ["machine8/results/jobs/ABL_noCL_prob_mc"], "no neg.": ["machine4/results/jobs/ABL_none_prob_mc"]}}

def shard_steps(S):
    """per shard: list of steps (video, k, start, success_time or None, feedback times) in order"""
    out = []
    for p in S:
        k = -1; cur = None
        for x, t in zip(p["pred_texts"], p["pred_timestamps"]):
            if x.startswith("Instruction"): k += 1; cur = dict(video=p["video_id"], k=k, instr=x.split(":", 1)[1].strip()[:80], start=t, succ=None, fb=[]); out.append(cur)
            elif x.startswith("Success"): cur["succ"] = t
            elif x.startswith("Feedback"): cur["fb"].append(t)
    return out

def parse_shard(steps, recs):
    """assign a shard's own record stream to its steps: list of per-step lists of (n, p_done, p_mis or None)"""
    res, i = [], 0
    for s in steps:
        checks = []
        while i < len(recs):
            r = recs[i]
            if r["mode"] != "instruction_end": return None
            if checks and r["n"] == 10: break                                   # next step starts
            if r["instr"] != s["instr"][:len(r["instr"])]: return None
            if r["yes"]: checks.append((r["n"], r["p"], None)); i += 1; break
            m = recs[i + 1]
            if m["mode"] != "mistake_inference": return None
            checks.append((r["n"], r["p"], m)); i += 2
        res.append(checks)
    return res if i == len(recs) else None

def demux(steps_by_shard, recs):
    """split an interleaved stream of two shards; where a record fits both shards, try both (depth-first) and keep the first
    assignment whose per-shard streams parse and reproduce the shard predictions (checked by the caller via `ok`)"""
    instr_seq = [[s["instr"] for s in st_] for st_ in steps_by_shard]; nsh = len(steps_by_shard)
    def fits(j, r, pos, last):
        if r["mode"] == "mistake_inference":
            return last[j] is not None and last[j]["mode"] == "instruction_end" and not last[j]["yes"] and last[j]["n"] == r["n"] and last[j]["instr"] == r["instr"]
        cur = instr_seq[j][pos[j]] if pos[j] < len(instr_seq[j]) else None; nxt = instr_seq[j][pos[j] + 1] if pos[j] + 1 < len(instr_seq[j]) else None
        same = cur is not None and cur.startswith(r["instr"]) and r["n"] != 10 and last[j] is not None and last[j]["mode"] == "mistake_inference"
        first = r["n"] == 10 and ((last[j] is None and cur is not None and cur.startswith(r["instr"])) or
                                  (last[j] is not None and (last[j]["mode"] == "mistake_inference" or last[j]["yes"]) and nxt is not None and nxt.startswith(r["instr"])))
        return same or first
    sys.setrecursionlimit(100000); tried = [0]
    def run(i, pos, last, assign):
        while i < len(recs):
            r = recs[i]; c = [j for j in range(nsh) if fits(j, r, pos, last)]
            if not c: return None
            if len(c) > 1:
                for j in c:
                    tried[0] += 1
                    if tried[0] > 64: return None
                    p2, l2 = list(pos), list(last)
                    if r["mode"] == "instruction_end" and r["n"] == 10 and l2[j] is not None: p2[j] += 1
                    l2[j] = r; out = run(i + 1, p2, l2, assign + [j])
                    if out is not None: return out
                return None
            j = c[0]
            if r["mode"] == "instruction_end" and r["n"] == 10 and last[j] is not None: pos = list(pos); pos[j] += 1
            last = list(last); last[j] = r; assign = assign + [j]; i += 1
        streams = [[] for _ in range(nsh)]
        for r, j in zip(recs, assign): streams[j].append(r)
        return streams if all(parse_and_check(st_, s_) for st_, s_ in zip(steps_by_shard, streams)) else None
    out = run(0, [0] * nsh, [None] * nsh, [])
    return out, tried[0]

def parse_and_check(steps, rec):
    parsed = parse_shard(steps, rec)
    if parsed is None: return False
    for s, checks in zip(steps, parsed):
        init = int(s["start"] * 2); succ = None; fb = []
        for j, (n, pd, m) in enumerate(checks, 1):
            t = (init + 10 * j) / 2
            if m is None: succ = t
            elif m["yes"]: fb.append(t)
        if succ != s["succ"] or [round(x, 3) for x in fb] != [round(x, 3) for x in s["fb"]]: return False
    return True

def load(d):
    shards = sorted(glob.glob(f"{d}/shard*.json")); S = [json.load(open(f)) for f in shards]; steps = [shard_steps(s) for s in S]
    per = [f"{d}/plog{f.rsplit('shard', 1)[1][:-5]}.jsonl" for f in shards]
    if all(os.path.exists(p) for p in per): streams = [[json.loads(l) for l in open(p)] for p in per]; amb = 0
    else:
        streams, amb = demux(steps, [json.loads(l) for l in open(f"{d}/plog.jsonl")])
        if streams is None: return None, f"demux failed (ambiguous {amb})"
    out = []
    for st_, rec in zip(steps, streams):
        parsed = parse_shard(st_, rec)
        if parsed is None: return None, "parse failed"
        for s, checks in zip(st_, parsed):
            init = int(s["start"] * 2); succ = None; fb = []
            for j, (n, pd, m) in enumerate(checks, 1):
                t = (init + 10 * j) / 2
                if m is None: succ = t
                elif m["yes"]: fb.append(t)
            if succ != s["succ"] or [round(x, 3) for x in fb] != [round(x, 3) for x in s["fb"]]: return None, f"decisions do not reproduce {s['video']} step {s['k']}"
            out.append(dict(video=s["video"], k=s["k"], start=s["start"], checks=[((init + 10 * j) / 2, pd) for j, (n, pd, m) in enumerate(checks, 1)]))
    return out, f"ok (ambiguous assignments {amb})"

def auroc(pos, neg):
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg]); r = 0; i = 0; R = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]: j += 1
        avg = (i + j + 1) / 2; R += avg * sum(1 for _, y in s[i:j] if y); i = j
    P, N = len(pos), len(neg); return (R - P * (P + 1) / 2) / max(P * N, 1)
res = {}
for dsn, groups in RUNS.items():
    ds = L.load("neurips2025" if dsn == "qicd" else "ego_mc_bench", "test"); GT = {}
    for a in ds.annotations:
        g = L.ev.get_ground_truth_data(ds, a["video_id"])
        for k, (ts, hm) in enumerate(zip(g["texts_timestamps"], g["has_mistake"])): GT[(a["video_id"], k)] = (ts[0], max(ts), hm)
    res[dsn] = {}
    for gname, paths in groups.items():
        for p in paths:
            data, msg = load(B + p); print(dsn, gname, p, msg, flush=True)
            if data is None: continue
            pts = []
            for s in data:
                st0, en, hm = GT[(s["video"], s["k"])]
                if hm or en - st0 < 10: continue
                for t, pd in s["checks"]: pts.append((s["video"], (t - st0) / (en - st0), pd))
            early = [x for x in pts if x[1] < 0.5]; late = [x for x in pts if 0.85 <= x[1] <= 1.0]
            r = dict(n_early=len(early), n_late=len(late), p_early=st.mean(x[2] for x in early), p_late=st.mean(x[2] for x in late),
                     auroc=auroc([x[2] for x in late], [x[2] for x in early]))
            vids = sorted({x[0] for x in pts}); rng = random.Random(0); bs = []
            byv = {v: ([x[2] for x in late if x[0] == v], [x[2] for x in early if x[0] == v]) for v in vids}
            for _ in range(500):
                smp = [rng.choice(vids) for _ in vids]; po = [y for v in smp for y in byv[v][0]]; ne = [y for v in smp for y in byv[v][1]]
                bs.append(auroc(po, ne))
            bs.sort(); r["auroc95"] = (bs[12], bs[487])
            res[dsn].setdefault(gname, []).append(dict(run=p, **r))
            print(f"   p_early {r['p_early']:.3f} (n {r['n_early']})  p_late {r['p_late']:.3f} (n {r['n_late']})  AUROC late-vs-early {r['auroc']:.3f} [{r['auroc95'][0]:.3f}, {r['auroc95'][1]:.3f}]", flush=True)
json.dump(res, open(f"{OUT}/progress_probe.json", "w"), indent=1)
