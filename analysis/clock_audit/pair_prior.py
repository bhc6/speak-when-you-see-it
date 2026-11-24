"""What a video-blind policy scores on the 931 time-matched QICD test pairs (supplement E). The pairs hold the elapsed time and the
window length fixed, so any policy of elapsed time alone ties every pair (0.5). A policy that also uses the instruction ranks the
two checks by elapsed time over the median annotated duration of their instruction in the 209 QICD training videos (validation
videos removed; global median if the instruction is unseen); ties count one half. Pairs and ECT seed 0 probabilities from the
pair probe (evidence_pair_probe.py output, which stores each pair's elapsed time and both instructions).
python3 pair_prior.py -> out/pair_prior.json"""
import json, os, statistics as st, collections
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(V0 + "analysis_code/blind_strong/inputs/gt_dump_1118.json"))
val = {g["video_id"] for g in G["neurips2025_validation"]}
norm = lambda s: s.replace("Instruction:", "").strip()
dur = collections.defaultdict(list); n_train = 0
for g in G["neurips2025_train"]:
    if g["video_id"] in val: continue
    n_train += 1
    for tx, ts in zip(g["instruction_segment_texts"], g["instruction_segment_texts_timestamps"]):
        if max(ts) > ts[0]: dur[norm(tx[0])].append(max(ts) - ts[0])
PR = {k: st.median(v) for k, v in dur.items()}; GM = st.median([x for v in dur.values() for x in v])
prior = lambda ins: PR.get(norm(ins), GM)
P = json.load(open(V0 + "machine5/results/jobs/epairs/T1v4_s0.json"))
acc = lambda L: sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in L) / len(L)
res = dict(n_train_videos=n_train)
for q, name in (("instruction_end", "completion"), ("mistake_inference", "mistake")):
    S = [p for p in P if p["q"] == q]
    res[name] = dict(pairs=len(S), same_instruction=sum(norm(p["pos_instr"]) == norm(p["neg_instr"]) for p in S),
                     elapsed_only=0.5, elapsed_over_prior=round(acc([(p["E"] / prior(p["pos_instr"]), p["E"] / prior(p["neg_instr"])) for p in S]), 3),
                     ect_s0=round(acc([(p["p_pos"], p["p_neg"]) for p in S]), 3))
    print(name, res[name])
json.dump(res, open(HERE + "/out/pair_prior.json", "w"), indent=1)
