"""Qwen2.5-VL (another model generation/interface: Qwen2_5_VLForConditionalGeneration, 28-px patches, fps-based
temporal ids) under the official turn-based baseline: same prompts (build_messages), same video windows (frames at 2 fps,
640x360 nearest, <=240 frames), same call sequence; greedy decoding. Combine with qic_online flags (--qic_mem).
Only model loading and the single-call inference function are replaced (Qwen2.5-VL takes fps instead of video_metadata)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import qic_greedy  # noqa: E402,F401  (dataset shard patch)
baseline = qic_greedy.baseline
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402


def _load(model_id, cache_dir=None):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    g = model.generation_config; g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    print(f"qic_q25: {type(model).__name__} greedy", flush=True)
    return model, AutoProcessor.from_pretrained(model_id)


@torch.no_grad()
def _ask(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    msgs = baseline.build_messages(mode, instruction=instruction)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[text], images=None, videos=[video], fps=2.0, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


baseline.load_model_and_processor = _load
baseline.get_qwen_vl_output = _ask
import qic_online  # noqa: E402,F401  (must come after the patch: it captures get_qwen_vl_output)
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
