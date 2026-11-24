"""Time-matched evidence pairs (Winoground / VQA-v2 complementary-pair style): each pair = two windows of the SAME
recording with the SAME elapsed time since step start (same number of frames) and the same official question, whose
correct answers differ. Completion pairs: a finished step (GT 'success' situation) vs another step of that video still
>= 20 s from its end. Mistake pairs: a GT mistake moment vs a mistake-free step at the same elapsed time, >= 15 s left.
Records first-token P(yes) for both members. Any policy that sees only elapsed time / window length scores both
members identically (pair accuracy 0.5 with ties = 0.5). Evaluation only; no tuning.
python evidence_pair_probe.py --frames_root FR --out OUT.json --split validation|test [--model_id M] [--lora DIR] [--frame_cap 32]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import numpy as np, torch
import qwen3_vl_baseline as B, sia_data, vlm_common as V
from utils import load_frames_into_array
ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct"); ap.add_argument("--lora", default=None)
ap.add_argument("--split", default="validation"); ap.add_argument("--black", action="store_true"); ap.add_argument("--frame_cap", type=int, default=32)
a = ap.parse_args()
from transformers import AutoProcessor
FAM = V.family(a.model_id)
model = V.model_class(FAM).from_pretrained(a.model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2" if FAM == "qwen3vl" else "sdpa", device_map="cuda:0").eval()
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
proc = V.fix_processor(AutoProcessor.from_pretrained(a.model_id), a.model_id, FAM); tok = proc.tokenizer
norm = lambda i: tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]")
yes = [i for i in range(len(tok)) if norm(i) == "yes"]; no = [i for i in range(len(tok)) if norm(i) == "no"]
@torch.no_grad()
def p_yes(mode, clip, instr):
    msgs = V.messages(B.build_messages(mode, instruction=instr), FAM)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = V.encode(proc, FAM, text, clip, a.frame_cap).to("cuda")
    for k, v in inp.items():
        if torch.is_floating_point(v): inp[k] = v.to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    p = torch.softmax(lg, -1); y, n = float(p[yes].sum()), float(p[no].sum())
    return round(y / max(y + n, 1e-9), 5)
_cache = {}
def frames(vid):
    if vid not in _cache:
        _cache.clear(); d = sia_data.frame_dir(a.frames_root, vid)
        fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
        _cache[vid] = load_frames_into_array([os.path.join(d, f) for f in fs], video_input_resolution=(640, 360))
    return _cache[vid]
# ---- time-matched evidence pairs (same recording, same elapsed time E, same window length, same question) ----
import random
ex, _ = sia_data.build_v2(a.frames_root, a.split)
steps = {}
for e in ex:
    if e["step"] >= 0: steps.setdefault(e["video_id"], set()).add((e["step"], e["start"], e["end"], e["instruction"]))
mis_steps = {(e["video_id"], e["step"]) for e in ex if e["kind"] == "mistake"}
rng = random.Random(2027); pairs = []
for e in sorted(ex, key=lambda e: (e["video_id"], e["t"], e["kind"])):
    E = e["t"] - e["start"]; S = sorted(steps.get(e["video_id"], []))
    if e["kind"] == "success":      # completion pair: finished step vs another step of the same video still >= 20 s from its end
        cand = [s for s in S if s[0] != e["step"] and s[2] - s[1] > E + 20]; q = "instruction_end"
    elif e["kind"] == "mistake":    # mistake pair: mistake moment vs a mistake-free step of the same video, >= 15 s left
        cand = [s for s in S if (e["video_id"], s[0]) not in mis_steps and s[2] - s[1] > E + 15]; q = "mistake_inference"
    else: continue
    if not cand: continue
    si, st, en, ins = rng.choice(cand)
    pairs.append(dict(video_id=e["video_id"], q=q, E=E, pos=dict(step=e["step"], start=e["start"], t=e["t"], instruction=e["instruction"]),
                      neg=dict(step=si, start=st, t=st + E, instruction=ins)))
done = {}
if os.path.exists(a.out):
    for r in json.load(open(a.out)): done[(r["video_id"], r["q"], r["pos_step"], round(r["E"], 3))] = r
out = list(done.values())
def clip_of(video, s):
    i0 = int(s["start"] * 2); seek = min(max(int(s["t"] * 2), i0 + 10), len(video)); return video[max(seek - 240, i0):seek]
for p in pairs:
    key = (p["video_id"], p["q"], p["pos"]["step"], round(p["E"], 3))
    if key in done: continue
    video = frames(p["video_id"]); cp, cn = clip_of(video, p["pos"]), clip_of(video, p["neg"])
    if a.black: cp, cn = np.zeros_like(cp), np.zeros_like(cn)
    out.append(dict(video_id=p["video_id"], q=p["q"], E=p["E"], pos_step=p["pos"]["step"], neg_step=p["neg"]["step"],
                    pos_instr=p["pos"]["instruction"], neg_instr=p["neg"]["instruction"], n_pos=len(cp), n_neg=len(cn),
                    p_pos=p_yes(p["q"], cp, p["pos"]["instruction"]), p_neg=p_yes(p["q"], cn, p["neg"]["instruction"])))
    if len(out) % 50 == 0: json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("evidence_pairs", len(out), "/", len(pairs), flush=True)
json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("evidence_pairs done", len(out), flush=True)
