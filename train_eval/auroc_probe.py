"""Discrimination check (no generation): for GT-sampled validation situations, the first-token P(yes) of the official
mistake prompt and completion prompt under a given model (zero-shot or LoRA). Same windows / prompts / preprocessing
as the official turn-based baseline (qwen3_vl_baseline.get_qwen_vl_output). Situations from sia_data.build_v2 on the
validation split (mistake at GT f, f+5, f+10; success at end, end+5; silence; shifted).
python auroc_probe.py --frames_root FR --out OUT.json [--lora DIR] [--ids a,b,c]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import torch
import qwen3_vl_baseline as B
if os.environ.get("QIC_INSTR_FIRST"):
    # prompt ablation: the step instruction is also given BEFORE the video, so the video tokens are encoded knowing
    # the step (causal LM). Everything else (system prompt, full official question after the video) is unchanged.
    _bm = B.build_messages
    def _bm_instr_first(mode, **kw):
        m = _bm(mode, **kw)
        m[1]["content"].insert(0, {"type": "text", "text": f"Current recipe step: {kw['instruction']}\n"})
        return m
    B.build_messages = _bm_instr_first
import sia_data
from utils import load_frames_into_array

ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--lora", default=None); ap.add_argument("--ids", default=None); ap.add_argument("--max_buffer", type=int, default=240)
a = ap.parse_args()
if os.environ.get("QIC_MEMFRAC"): torch.cuda.set_per_process_memory_fraction(float(os.environ["QIC_MEMFRAC"]))
model, proc = B.load_model_and_processor("Qwen/Qwen3-VL-2B-Instruct")
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
tok = proc.tokenizer
yes = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "yes"]
no = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "no"]
ids = set(a.ids.split(",")) if a.ids else None
ex, _ = sia_data.build_v2(a.frames_root, "validation", ids=ids)


@torch.no_grad()
def p_yes(mode, clip, instr):
    text = proc.apply_chat_template(B.build_messages(mode, instruction=instr), tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=None, videos=clip, padding=True, video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt")
    inp = inp.to("cuda").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    p = torch.softmax(lg, -1); y, n = float(p[yes].sum()), float(p[no].sum())
    return y / max(y + n, 1e-9)


out, cur, video = [], None, None
for e in sorted(ex, key=lambda e: e["video_id"]):
    if e["video_id"] != cur:
        cur = e["video_id"]; d = sia_data.frame_dir(a.frames_root, cur)
        fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
        video = load_frames_into_array([os.path.join(d, f) for f in fs], video_input_resolution=(640, 360))
    i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(video)); b0 = max(seek - a.max_buffer, i0)
    clip = video[b0:seek]
    out.append({"video_id": cur, "step": e["step"], "kind": e["kind"], "t": e["t"], "elapsed": e["t"] - e["start"],
                "p_mis": p_yes("mistake_inference", clip, e["instruction"]), "p_done": p_yes("instruction_end", clip, e["instruction"])})
    if len(out) % 100 == 0:
        json.dump(out, open(a.out, "w")); print("auroc_probe", len(out), "/", len(ex), flush=True)
json.dump(out, open(a.out, "w")); print("auroc_probe done", len(out), flush=True)
