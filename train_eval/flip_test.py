"""Shortcut test: does a model's P(yes | mistake prompt) depend on the instruction's surface form ('Now ...' vs not)?
Validation subset situations (mistake / clean / success / silence), each scored twice with the official mistake prompt:
original QICD instruction ('Now add ...') and the same text with the leading 'Now ' removed and re-capitalised (CoMist
style). Same frames, same windows. python flip_test.py --frames_root FR --ids IDS --out OUT.json [--lora DIR]"""
import argparse, json, os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import torch
import qwen3_vl_baseline as B
import sia_data
from utils import load_frames_into_array
ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ids", required=True); ap.add_argument("--lora", default=None); ap.add_argument("--per_kind", type=int, default=60)
a = ap.parse_args()
model, proc = B.load_model_and_processor("Qwen/Qwen3-VL-2B-Instruct")
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
tok = proc.tokenizer
YES = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "yes"]
NO = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]") == "no"]
ex, _ = sia_data.build_v3(a.frames_root, "validation", ids=set(a.ids.split(",")), comist=False)
rng = random.Random(0); pick = []
for k in ("mistake", "clean", "success", "silence"):
    L = [e for e in ex if e["kind"] == k]; pick += rng.sample(L, min(a.per_kind, len(L)))
def denow(s):
    s = s[4:] if s.startswith("Now ") else s
    return s[:1].upper() + s[1:]
@torch.no_grad()
def p_yes(clip, instr):
    text = proc.apply_chat_template(B.build_messages("mistake_inference", instruction=instr), tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=None, videos=clip, padding=True, video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt").to("cuda").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
    y, n = float(p[YES].sum()), float(p[NO].sum()); return y / max(y + n, 1e-9)
out, cur = [], None
for e in sorted(pick, key=lambda e: e["video_id"]):
    if e["video_id"] != cur:
        cur = e["video_id"]; d = sia_data.frame_dir(a.frames_root, cur)
        fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
        video = load_frames_into_array([os.path.join(d, f) for f in fs], video_input_resolution=(640, 360))
    i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(video)); clip = video[max(seek - 240, i0):seek]
    out.append({"video_id": cur, "kind": e["kind"], "instr": e["instruction"], "p_orig": p_yes(clip, e["instruction"]), "p_denow": p_yes(clip, denow(e["instruction"]))})
json.dump(out, open(a.out, "w")); print("flip done", len(out), flush=True)
