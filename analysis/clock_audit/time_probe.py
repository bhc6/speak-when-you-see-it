"""Clock-only probe (validation only, no generation; Qwen3-VL only). Same fixed situations and first-token P(yes)/(P(yes)+P(no))
as grounding_probe.py (sia_data.build_v2, --stride 3 = the 1,233 situations of the paper's probe), under three conditions
that change ONLY the time stamps the processor writes between frames; pixels, frame count and prompt are identical
(checked on the evaluation machine: pixel_values_videos and video_grid_thw equal for fps 1 / 2 / 4 with do_sample_frames=False):
  ts1  : video_metadata fps=2 (the official setting; reproduces grounding_probe 'orig')
  ts2  : fps=1   -> every time stamp doubled (the window reads as twice as long)
  ts05 : fps=4   -> every time stamp halved
--black replaces every frame by black (for the model trained on black frames; the time stamps are then its only clock
besides the frame count).
python time_probe.py --frames_root FR --out OUT.json [--lora DIR] [--stride 3] [--black]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import numpy as np, torch
import qwen3_vl_baseline as B, sia_data, vlm_common as V
from utils import load_frames_into_array
ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct"); ap.add_argument("--lora", default=None)
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--black", action="store_true")
a = ap.parse_args()
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:0").eval()
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
proc = V.fix_processor(AutoProcessor.from_pretrained(a.model_id), a.model_id, "qwen3vl"); tok = proc.tokenizer
norm = lambda i: tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]")
yes = [i for i in range(len(tok)) if norm(i) == "yes"]; no = [i for i in range(len(tok)) if norm(i) == "no"]
ex, _ = sia_data.build_v2(a.frames_root, "validation"); ex = sorted(ex, key=lambda e: (e["video_id"], e["t"]))[::a.stride]
FPS = {"ts1": 2, "ts2": 1, "ts05": 4}


@torch.no_grad()
def p_yes(mode, clip, instr, fps):
    msgs = V.messages(B.build_messages(mode, instruction=instr), "qwen3vl")
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=None, videos=clip, padding=True, do_sample_frames=False, return_tensors="pt",
               video_metadata=[{"fps": fps, "total_num_frames": len(clip), "frames_indices": list(range(len(clip)))}]).to("cuda")
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
out = list(done.values())
for e in ex:
    key = (e["video_id"], e["step"], e["kind"], e["t"])
    if key in done: continue
    video = frames(e["video_id"])
    i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(video)); b0 = max(seek - 240, i0); clip = video[b0:seek]
    if a.black: clip = np.zeros_like(clip)
    r = {"video_id": e["video_id"], "step": e["step"], "kind": e["kind"], "t": e["t"], "elapsed": e["t"] - e["start"], "n": len(clip)}
    for c, fps in FPS.items():
        r[f"done_{c}"] = p_yes("instruction_end", clip, e["instruction"], fps); r[f"mis_{c}"] = p_yes("mistake_inference", clip, e["instruction"], fps)
    out.append(r)
    if len(out) % 50 == 0: json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("time_probe", len(out), "/", len(ex), flush=True)
json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out); print("time_probe done", len(out), flush=True)
