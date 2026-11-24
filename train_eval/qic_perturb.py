"""Visual-perturbation ladder for the official per-step protocol (validation analysis; Qwen3-VL + optional LoRA).
Only the clip handed to the model at each check is changed; GT restarts, 5-s checks, window length, prompts, parsing
and greedy decoding are unchanged (so window length / elapsed time stay available - this isolates the image content):
  black   : every frame replaced by a black frame (same count)  -> the model's own video-blind version
  static  : every frame replaced by the LAST frame of the window (current state only, no motion)
  shuffle : frames of the window in a fixed random order (seeded by clip end index) -> content kept, order destroyed
  capN    : uniform temporal subsampling to at most N frames, last frame kept (frame-budget control for other backbones)
    QIC_PERTURB=black [QIC_LORA=dir] python qic_perturb.py <baseline args>   (also --perturb MODE / --qic_lora DIR)"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "--perturb" in sys.argv:
    k = sys.argv.index("--perturb"); os.environ["QIC_PERTURB"] = sys.argv[k + 1]; del sys.argv[k:k + 2]
if "--qic_lora" in sys.argv or os.environ.get("QIC_LORA"):
    import qic_lora as _w  # noqa: E402
else:
    import qic_greedy as _w  # noqa: E402
baseline = _w.baseline; MODE = os.environ["QIC_PERTURB"]; _ask = baseline.get_qwen_vl_output
assert MODE in ("black", "static", "shuffle") or MODE.startswith("cap"), MODE

def _perturb(v):
    if MODE == "black": return np.zeros_like(v)
    if MODE == "static": return np.repeat(v[-1:], len(v), axis=0)
    if MODE.startswith("cap"):
        import vlm_common as V
        return V.subsample(v, int(MODE[3:]))
    end = (v.__array_interface__["data"][0] - v.base.__array_interface__["data"][0]) // v.base.strides[0] + len(v) if v.base is not None else len(v)
    return v[np.random.default_rng(int(end)).permutation(len(v))]

def _get(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    return _ask(mode, model, processor, video=_perturb(video), instruction=instruction, max_new_tokens=max_new_tokens)

baseline.get_qwen_vl_output = _get
print(f"qic_perturb: mode={MODE}", flush=True)
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
