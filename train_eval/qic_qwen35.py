"""qic_shard.py with a Qwen3.5 model in place of Qwen3-VL (run with PYTHONPATH=WORKSPACE/qic/py35: transformers 5).

The baseline hard-codes Qwen3VLForConditionalGeneration; this wrapper swaps in AutoModelForImageTextToText (via _Auto) so the
same load call (flash_attention_2, bfloat16) builds Qwen3_5ForConditionalGeneration. Qwen3.5 uses the same
Qwen3VLProcessor/video processor, so prompts, frames and windows are unchanged. Qwen3.5-2B ships no
generation_config.json, so generation is greedy unless QIC_SAMPLE=1 (Qwen's recommended non-thinking VL sampling,
without presence_penalty, which HF generate lacks). Qwen3.5-2B is non-thinking by default.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qic_shard  # noqa: E402  (patches the baseline's dataset class)
from transformers import AutoModelForImageTextToText  # noqa: E402

baseline = qic_shard.baseline


class _Auto:
    """from_pretrained only; transformers 5 renamed torch_dtype and sets no hf_device_map for a single device."""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        if "torch_dtype" in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = AutoModelForImageTextToText.from_pretrained(*args, **kwargs)
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {"": kwargs.get("device_map")}
        return model


baseline.Qwen3VLForConditionalGeneration = _Auto
_load = baseline.load_model_and_processor


def _load35(*args, **kwargs):
    model, processor = _load(*args, **kwargs)
    g = model.generation_config
    if os.environ.get("QIC_SAMPLE") == "1":
        g.do_sample, g.temperature, g.top_p, g.top_k = True, 0.7, 0.8, 20
    else:
        g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    print(f"qic_qwen35: {type(model).__name__} attn={model.config._attn_implementation} {g}", flush=True)
    return model, processor


baseline.load_model_and_processor = _load35

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
