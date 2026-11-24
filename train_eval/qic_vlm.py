"""Official turn-based per-step protocol with OTHER VLM families (InternVL3.5-HF, LLaVA-OneVision-HF), optional LoRA.
Unchanged from the official baseline: dataset, GT restarts, 5-s checks, window [step start, t] (<=240 frames at 2 fps,
640x360), prompts (baseline.build_messages), 'yes' parsing, greedy decoding. Changed (disclosed): the model/processor,
and the clip is uniformly subsampled to at most QIC_FRAME_CAP frames (default 32; last frame kept) because these models
spend 196-256 tokens per frame. Exactly the same subsampling is used in training (vlm_common).
    QIC_FRAME_CAP=32 [QIC_LORA=dir] python qic_vlm.py <baseline args> --model_id OpenGVLab/InternVL3_5-8B-HF"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for flag, env in (("--qic_lora", "QIC_LORA"), ("--frame_cap", "QIC_FRAME_CAP")):
    if flag in sys.argv:
        k = sys.argv.index(flag); os.environ[env] = sys.argv[k + 1]; del sys.argv[k:k + 2]
import torch  # noqa: E402
import qic_shard  # noqa: E402  (dataset sharding / QIC_SKIP / QIC_SWAP)
import vlm_common as V  # noqa: E402
baseline = qic_shard.baseline
CAP = int(os.environ.get("QIC_FRAME_CAP", "32")); STATE = {}

def _load(model_id, cache_dir=None):
    from transformers import AutoProcessor
    fam = V.family(model_id); C = V.model_class(fam)
    model = C.from_pretrained(model_id, cache_dir=cache_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0").eval()
    if os.environ.get("QIC_LORA"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ["QIC_LORA"]).merge_and_unload().eval()
        print(f"qic_vlm: merged {os.environ['QIC_LORA']}", flush=True)
    g = model.generation_config; g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    proc = V.fix_processor(AutoProcessor.from_pretrained(model_id), model_id, fam); STATE.update(fam=fam)
    print(f"qic_vlm: {model_id} family={fam} frame_cap={CAP} greedy", flush=True)
    return model, proc

@torch.no_grad()
def _ask(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    msgs = V.messages(baseline.build_messages(mode, instruction=instruction), STATE["fam"])
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = V.encode(processor, STATE["fam"], text, video, CAP).to("cuda")
    for k, v in inp.items():
        if torch.is_floating_point(v): inp[k] = v.to(torch.bfloat16)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

baseline.load_model_and_processor = _load
baseline.get_qwen_vl_output = _ask
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
