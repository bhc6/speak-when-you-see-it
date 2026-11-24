"""Geometry probe: is SIA actually pulling situations toward their intervention, and does it generalise?
Validation subset situations from sia_data.build_v3 (validation: mistake f/f+5/f+10, clean = elapsed-matched mistake-free
negatives, success end/end+5, notdone = elapsed-matched, silence, shifted). For each model, one forward per situation
with the official MISTAKE prompt (same windows/preprocessing as the baseline); stores last-prompt-token states at layers
7/14/21/28. With --head CKPT (sia_train last.pt of the same step as --lora) also stores the SIA head output g_s and the
prototype logits. python geom_probe.py --frames_root FR --ids IDS --out OUT [--lora DIR] [--head last.pt]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "..", "ref_sys", "ai_coach_cooking_2026")))
import numpy as np, torch, torch.nn.functional as F
import qwen3_vl_baseline as B
import sia_data
from utils import load_frames_into_array

ap = argparse.ArgumentParser(); ap.add_argument("--frames_root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ids", default=None); ap.add_argument("--lora", default=None); ap.add_argument("--head", default=None)
ap.add_argument("--shard", default="0/1"); ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
DEV = "cpu" if a.cpu else "cuda"
if os.environ.get("QIC_MEMFRAC"): torch.cuda.set_per_process_memory_fraction(float(os.environ["QIC_MEMFRAC"]))
if a.cpu:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(int(os.environ.get("NT", "16")))
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu").eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
else:
    model, proc = B.load_model_and_processor("Qwen/Qwen3-VL-2B-Instruct")
if a.lora:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
LAY = [7, 14, 21, 28]
head = None
if a.head:
    st = torch.load(a.head, map_location=DEV, weights_only=False); hs = st["hs"]
    W1, b1, W2, b2 = hs["net.0.weight"], hs["net.0.bias"], hs["net.2.weight"], hs["net.2.bias"]
    mu, sd, P = hs["mu"], hs["sd"], F.normalize(st["protos"].float(), dim=-1)
    head = lambda x: F.normalize(F.linear(F.gelu(F.linear((x - mu) / sd, W1.float(), b1.float())), W2.float(), b2.float()), dim=-1)
ex, _ = sia_data.build_v3(a.frames_root, "validation", ids=set(a.ids.split(",")) if a.ids else None, comist=False)
_si, _sn = map(int, a.shard.split("/")); vids = sorted({e["video_id"] for e in ex}); keep = set(vids[_si::_sn])
ex = [e for e in ex if e["video_id"] in keep]
H, G, meta, cur = [], [], [], None
for e in sorted(ex, key=lambda e: e["video_id"]):
    if e["video_id"] != cur:
        cur = e["video_id"]; d = sia_data.frame_dir(a.frames_root, cur)
        fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
        video = load_frames_into_array([os.path.join(d, f) for f in fs], video_input_resolution=(640, 360))
    i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(video)); b0 = max(seek - 240, i0)
    text = proc.apply_chat_template(B.build_messages("mistake_inference", instruction=e["instruction"]), tokenize=False, add_generation_prompt=True)
    clip = video[b0:seek]
    inp = proc(text=[text], images=None, videos=clip, padding=True, video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt").to(DEV).to(torch.bfloat16)
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        hsd = model(**inp, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
    H.append(np.stack([hsd[L][0, -1].float().cpu().numpy() for L in LAY]).astype(np.float16))
    if head is not None:
        with torch.no_grad():
            g = head(hsd[14][0, -1].float()); G.append(np.concatenate([g.cpu().numpy(), (P @ g).cpu().numpy()]))
    meta.append({"video_id": cur, "step": e["step"], "kind": e["kind"], "elapsed": e["t"] - e["start"], "cat": e.get("cat")})
    if len(meta) % 200 == 0: print("geom", len(meta), "/", len(ex), flush=True)
np.savez(a.out + ".npz", h=np.stack(H), g=np.stack(G) if G else np.zeros(0)); json.dump(meta, open(a.out + ".json", "w"))
print("geom done", len(meta), flush=True)
