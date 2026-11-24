"""Grounding plane probe (validation only, no generation): first-token P(yes)/(P(yes)+P(no)) of the official completion and
mistake prompts on FIXED GT-sampled validation situations (sia_data.build_v2, same as auroc_probe) under 5 conditions:
  orig   : official window [step start, t] (<=240 frames)
  black  : same number of frames, all black            -> evidence removed, elapsed-time cue kept
  static : every frame = last frame of the window       -> only the current state
  pad60  : 120 copies of the window's FIRST frame prepended (cap 240, cut from the front) -> elapsed cue +60 s, content kept
  swap   : same frame indices taken from another validation recording of the same recipe (swap_val.json)
Visual sensitivity = mean |p_orig - p_swap| (and black); time sensitivity = mean |p_orig - p_pad60|.
Works for Qwen3-VL and the other families through vlm_common (32-frame cap for non-Qwen).
python grounding_probe.py --frames_root FR --out OUT.json [--model_id M] [--lora DIR] [--stride 2] [--frame_cap 32]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import numpy as np, torch
import qwen3_vl_baseline as B, sia_data, vlm_common as V
from utils import load_frames_into_array
ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct"); ap.add_argument("--lora", default=None)
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--frame_cap", type=int, default=32)
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
ex, _ = sia_data.build_v2(a.frames_root, "validation"); ex = sorted(ex, key=lambda e: (e["video_id"], e["t"]))[::a.stride]
SW = json.load(open(os.path.join(HERE, "swap_val.json")))
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
done = {}
if os.path.exists(a.out):
    for r in json.load(open(a.out)): done[(r["video_id"], r["step"], r["kind"], r["t"])] = r
out = list(done.values()); donor_cache = {}
for e in ex:
    key = (e["video_id"], e["step"], e["kind"], e["t"])
    if key in done: continue
    video = frames(e["video_id"])
    i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(video)); b0 = max(seek - 240, i0); clip = video[b0:seek]
    dv = SW.get(e["video_id"])
    if dv not in donor_cache:
        donor_cache.clear(); d = sia_data.frame_dir(a.frames_root, dv)
        fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
        donor_cache[dv] = [os.path.join(d, f) for f in fs]
    dp = donor_cache[dv]; idx = [min(i, len(dp) - 1) for i in range(b0, seek)]
    conds = {"orig": clip, "black": np.zeros_like(clip), "static": np.repeat(clip[-1:], len(clip), 0),
             "pad60": np.concatenate([np.repeat(clip[:1], 120, 0), clip])[-240:],
             "swap": load_frames_into_array([dp[i] for i in idx], video_input_resolution=(640, 360))}
    r = {"video_id": e["video_id"], "step": e["step"], "kind": e["kind"], "t": e["t"], "elapsed": e["t"] - e["start"], "n": len(clip)}
    for c, cl in conds.items():
        r[f"done_{c}"] = p_yes("instruction_end", cl, e["instruction"]); r[f"mis_{c}"] = p_yes("mistake_inference", cl, e["instruction"])
    out.append(r)
    if len(out) % 50 == 0: json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("grounding_probe", len(out), "/", len(ex), flush=True)
json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("grounding_probe done", len(out), flush=True)
