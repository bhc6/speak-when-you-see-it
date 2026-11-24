"""Shared helpers for non-Qwen3-VL backbones (InternVL3.5-HF, LLaVA-OneVision-HF) used by BOTH training
(sia_train.py) and evaluation (qic_vlm.py), so that the two see identical inputs.
- family(model_id) -> "qwen3vl" | "internvl" | "llava_ov"
- subsample(clip, cap): uniform temporal subsampling to at most `cap` frames, always keeping the first and the LAST
  frame (the current moment). cap <= 0 keeps all frames.
- encode(processor, family, text, clip): processor call with the same 640x360 frames the official baseline loads."""
import numpy as np

def family(model_id):
    m = model_id.lower()
    if "qwen3-vl" in m: return "qwen3vl"
    if "internvl" in m: return "internvl"
    if "llava-onevision" in m or "llava_onevision" in m: return "llava_ov"
    raise ValueError(f"unsupported model {model_id}")

def subsample(clip, cap):
    n = len(clip)
    if cap <= 0 or n <= cap: return clip
    idx = np.unique(np.linspace(0, n - 1, cap).round().astype(int))
    return clip[idx]

def model_class(fam):
    if fam == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration as C
    elif fam == "internvl":
        from transformers import InternVLForConditionalGeneration as C
    else:
        from transformers import LlavaOnevisionForConditionalGeneration as C
    return C

def encode(processor, fam, text, clip, cap):
    if fam == "qwen3vl":
        return processor(text=[text], images=None, videos=clip, padding=True,
                         video_metadata=[{"fps": 2, "total_num_frames": len(clip)}], return_tensors="pt")
    v = subsample(clip, cap)
    return processor(text=[text], videos=[v], padding=True, return_tensors="pt")

def vision_module(model):
    m = model
    for attr in ("base_model", "model"):
        while hasattr(m, attr) and not hasattr(m, "vision_tower") and not hasattr(m, "visual"):
            m = getattr(m, attr)
    return getattr(m, "visual", None) or getattr(m, "vision_tower", None)

def messages(msgs, fam):
    """Some chat templates (LLaVA-OneVision) silently drop a system message whose content is a plain string; give every
    message list-of-parts content for non-Qwen3-VL families (the official baseline's text is unchanged)."""
    if fam == "qwen3vl": return msgs
    out = []
    for m in msgs:
        c = m["content"]
        out.append(dict(m, content=[{"type": "text", "text": c}] if isinstance(c, str) else c))
    return out

def fix_processor(proc, model_id, fam):
    """InternVL3.5-HF ships a video processor with size 384 while the vision tower expects 448 (image processor and
    config say 448) -> the pixel-shuffle reshape fails. Align the video processor with the model."""
    if fam == "internvl":
        from transformers import AutoConfig
        h, w = AutoConfig.from_pretrained(model_id).vision_config.image_size
        proc.video_processor.size = {"height": h, "width": w}
    return proc
