"""Phase A of situation-intervention alignment: frozen-VLM probe (no training, no generation).

Situations are sampled from GT (diagnostic, validation split): for every GT step segment
  mistake  at each GT mistake-feedback timestamp            -> target = that feedback text
  success  at the GT end of every mistake-free step         -> target = "Success"
  silence  every 20 s from step start + 10 s, kept only if >= 15 s from any GT feedback of the step and
           <= GT end - 15 s                                  -> target = silence
The window is exactly the turn-based baseline's: frames from the step's GT start up to t, capped at
--max_buffer_size (240 = 120 s), at least --video_seek_amount frames. The VLM sees the baseline's system text and the
step instruction with the neutral question below, and one forward pass stores last-prompt-token hidden states of the
layers in LAYERS (float16). Candidate intervention texts are encoded separately (text only, see probe_text.py).
Output: <save_file> (json list per video: items with type/t/step/text) and <save_file>.npz (hidden states, rows in
item order). Video selection via qic_shard (QIC_SHARD/QIC_SKIP).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import qic_shard  # noqa: E402
import torch  # noqa: E402

if any("Qwen3.5" in a for a in sys.argv):
    import qic_qwen35  # noqa: E402,F401  (model-class swap; needs PYTHONPATH=py35)
if os.environ.get("QIC_MEMFRAC"):  # cap this process so a concurrent job keeps its GPU memory
    torch.cuda.set_per_process_memory_fraction(float(os.environ["QIC_MEMFRAC"]))
baseline = qic_shard.baseline
SYSTEM = "You are an expert cooking assistant who is observing a person cook. "
QUESTION = ("The person is currently at the following recipe step: {instruction} "
            "What should you say to the person right now? If nothing needs to be said, stay silent.")
LAYERS = [7, 14, 21, 28]  # Qwen3-VL-2B (28 layers): 1/4, 1/2, 3/4, last
if "--qic_layers" in sys.argv:  # same relative depths for other models, e.g. Qwen3.5-2B (24 layers): 6,12,18,24
    k = sys.argv.index("--qic_layers"); LAYERS = [int(x) for x in sys.argv[k + 1].split(",")]; del sys.argv[k:k + 2]


def sample_items(d):
    items = []
    for si in range(d["num_of_instruction_segments"]):
        texts, ts, ty = d["gt_texts"][si], d["gt_text_timestamps"][si], d["gt_text_types"][si]
        start, end = ts[0], max(ts)
        fb_t = [t for t, y in zip(ts[1:], ty[1:]) if "feedback" in y]
        for x, t, y in zip(texts[1:], ts[1:], ty[1:]):
            if "mistake" in y:
                items.append({"type": "mistake", "t": t, "step": si, "text": x, "gt_type": y})
        if not d["gt_has_mistake"][si]:
            items.append({"type": "success", "t": end, "step": si, "text": "Success", "gt_type": "success"})
        t = start + 10
        while t <= end - 15:
            if all(abs(t - f) >= 15 for f in fb_t):
                items.append({"type": "silence", "t": t, "step": si, "text": "", "gt_type": "silence"})
            t += 20
    return items


@torch.no_grad()
def encode(model, processor, clip, instruction):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION.format(instruction=instruction)}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=clip, padding=True,
                       video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt")
    inputs = inputs.to("cuda").to(torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return np.stack([out.hidden_states[L][0, -1].float().cpu().numpy() for L in LAYERS]).astype(np.float16)


def run(args):
    res = (args.video_input_width, args.video_input_height)
    path = os.path.join(args.save_root, args.save_file); os.makedirs(args.save_root, exist_ok=True)
    ds = baseline.QualcommInteractiveCookingDatasetVideos(captaincook4d_root=args.captaincook4d_root,
                                                          plan_set=args.plan_set, split=args.split, model_fps=args.model_fps)
    model, processor = baseline.load_model_and_processor(args.model_id, args.cache_dir)
    print(f"qic_probe: layers {LAYERS} max_buffer {args.max_buffer_size}", flush=True)
    saved, H = [], []
    for vi in range(len(ds)):
        d = ds[vi]; ts = d["video_frame_timestamps"]; n = len(ts)
        video = baseline.load_frames_into_array(d["video_frame_paths"], video_input_resolution=res)
        items = sample_items(d)
        for it in items:
            s = d["gt_texts"][it["step"]]; start = d["gt_text_timestamps"][it["step"]][0]
            i0 = int(start * args.video_fps)
            seek = min(max(int(it["t"] * args.video_fps), i0 + args.video_seek_amount), n)
            b0 = max(seek - args.max_buffer_size, i0)
            it.update(buf_start=b0, seek=seek, instruction=s[0])
            H.append(encode(model, processor, video[b0:seek], s[0]))
        saved.append({"video_id": d["video_id"], "items": items})
        json.dump(saved, open(path, "w")); np.savez(path + ".npz", h=np.stack(H) if H else np.zeros((0,)))
        print(f"qic_probe: {d['video_id']} items {len(items)} total {len(H)}", flush=True)


if __name__ == "__main__":
    run(baseline.parse_args())
