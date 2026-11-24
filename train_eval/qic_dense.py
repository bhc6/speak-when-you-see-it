"""Dense probability log for exact offline replay of the official turn-based baseline under any decision thresholds.

In the turn-based protocol each step restarts from its GT start and every check is a pure function of
(step, seek index): window = frames[max(seek-240, start) : seek], prompts and preprocessing as in
qwen3_vl_baseline.get_qwen_vl_output (no session memory). So logging the first-token P(yes)/(P(yes)+P(no)) of both
official questions at every check lets dense_sim.py replay the baseline loop for any (tau_done, tau_mis, order)
exactly, as long as the replay never needs a check that was not logged. Per step we log checks until
  stop : p_done >= STOP and p_mis < MIS_LO=0.5 (every policy with tau_done <= STOP=0.9 and tau_mis >= MIS_LO ends here or earlier)
  cap  : check time > GT end + CAP s (replay beyond this is unknown and is counted by dense_sim.py)
  end  : end of video (exact: the baseline also stops here).
python qic_dense.py --captaincook4d_root R --split validation --out OUT.jsonl [--lora DIR] [--stop 0.9 --mis_lo 0.3 --cap 60]
Sharding: QIC_SHARD=i/n (by video index). Resumable: videos already in OUT.jsonl are skipped."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import torch
import qwen3_vl_baseline as B
from data import QualcommInteractiveCookingDatasetVideos
from utils import load_frames_into_array

ap = argparse.ArgumentParser()
ap.add_argument("--captaincook4d_root", required=True); ap.add_argument("--split", default="validation")
ap.add_argument("--plan_set", default="advanced_planning"); ap.add_argument("--out", required=True); ap.add_argument("--lora")
ap.add_argument("--stop", type=float, default=0.9); ap.add_argument("--mis_lo", type=float, default=0.5); ap.add_argument("--cap", type=float, default=60.0)
ap.add_argument("--sia_ckpt", help="also log SIA type probabilities (sia_score.SIAScorer; checks become [seek,t,pd,pm,sd,sm])")
ap.add_argument("--stop_sia", type=float, default=None, help="completion stop level for SIA scores (both_done); default --stop")
ap.add_argument("--bank", help="train feedback bank json (with --sia_ckpt)")
ap.add_argument("--stop_score", default="lm", choices=["lm", "sia", "both", "both_done", "sia_done"], help="which scores the early-stop rule uses (both = stop only when LM and SIA both satisfy it: replay exact for either)")
a = ap.parse_args()
FPS, SEEK, BUF = 2, 10, 240
if os.environ.get("QIC_MEMFRAC"): torch.cuda.set_per_process_memory_fraction(float(os.environ["QIC_MEMFRAC"]))
model, proc = B.load_model_and_processor("Qwen/Qwen3-VL-2B-Instruct")
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
tok = proc.tokenizer
YES = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "yes"]
NO = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "no"]


SIA = None
if a.sia_ckpt:
    import sia_score
    SIA = sia_score.SIAScorer(model, proc, a.sia_ckpt, a.bank)
LAST = {}


@torch.no_grad()
def p_yes(mode, clip, instr):
    text = proc.apply_chat_template(B.build_messages(mode, instruction=instr), tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=None, videos=clip, padding=True, video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt")
    inp = inp.to("cuda").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(**inp, use_cache=False, logits_to_keep=1, output_hidden_states=SIA is not None)
        lg = o.logits[0, -1].float()
    if SIA is not None:
        LAST[mode] = SIA.score(o.hidden_states)
    p = torch.softmax(lg, -1); y, n = float(p[YES].sum()), float(p[NO].sum())
    return y / max(y + n, 1e-9)


ds = QualcommInteractiveCookingDatasetVideos(captaincook4d_root=a.captaincook4d_root, plan_set=a.plan_set, split=a.split, model_fps=FPS)
si, sn = map(int, os.environ.get("QIC_SHARD", "0/1").split("/"))
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["video_id"] for l in open(a.out) if l.strip()}
for idx in (reversed(range(len(ds))) if os.environ.get("QIC_REVERSE") else range(len(ds))):
    if idx % sn != si: continue
    data = ds[idx]; vid = data["video_id"]
    if vid in done: continue
    if os.environ.get("QIC_ONLY") and vid not in os.environ["QIC_ONLY"].split(","): continue
    video = load_frames_into_array(data["video_frame_paths"], video_input_resolution=(640, 360))
    fts = data["video_frame_timestamps"]; n = len(data["video_frame_paths"])
    steps = []
    for k in range(data["num_of_instruction_segments"]):
        instr = data["gt_texts"][k][0]; start = data["gt_text_timestamps"][k][0]; end = data["gt_text_timestamps"][k][-1]
        init = int(start * FPS); seek = init + SEEK; checks, why = [], "end"; seen_lm = seen_sia = False
        while seek < n:
            if fts[seek] > end + a.cap: why = "cap"; break
            clip = video[max(seek - BUF, init):seek]
            pd, pm = p_yes("instruction_end", clip, instr), p_yes("mistake_inference", clip, instr)
            row = [seek, round(fts[seek], 3), round(pd, 5), round(pm, 5)]
            if SIA is not None:
                sd, sm = LAST["instruction_end"]["p_success"], LAST["mistake_inference"]["p_mistake"]
                row += [round(sd, 5), round(sm, 5)]
            checks.append(row)
            seek += SEEK
            ok = lambda qd, qm: qd >= a.stop and qm < a.mis_lo - 1e-5
            if (a.stop_score == "lm" and ok(row[2], row[3])) or (a.stop_score == "sia" and ok(row[4], row[5])) or \
               (a.stop_score == "both" and ok(row[2], row[3]) and ok(row[4], row[5])): why = "stop"; break
            if a.stop_score == "sia_done":     # only SIA decisions are replayed (exact for done_first, tau_done <= stop_sia)
                if row[4] >= (a.stop_sia if a.stop_sia is not None else a.stop): why = "stop"; break
            if a.stop_score == "both_done":
                # done_first replay is exact for tau_done <= the stop level of each score once that score has reached it
                # (completion is asked first); mistake_first rows that run past the log are flagged by dense_sim (caps)
                seen_lm |= row[2] >= a.stop; seen_sia |= row[4] >= (a.stop_sia if a.stop_sia is not None else a.stop)
                if seen_lm and seen_sia: why = "stop"; break
        steps.append({"k": k, "instruction": instr, "start": start, "end": end, "checks": checks, "why": why})
    with open(a.out, "a") as f:
        f.write(json.dumps({"video_id": vid, "n_frames": n, "global_start": data["global_start_timestamp"], "fts0": fts[0],
                            "steps": steps, "stop": a.stop, "stop_sia": a.stop_sia, "stop_score": a.stop_score, "mis_lo": a.mis_lo, "cap": a.cap}) + "\n")
    print("dense", vid, sum(len(s["checks"]) for s in steps), "checks", flush=True)
print("dense done", flush=True)
