"""qic_shard.py with greedy decoding: the Qwen3-VL generation_config samples (do_sample=True, temperature 0.7,
top_p 0.8, top_k 20) and the baseline sets no seed; this wrapper only turns sampling off after the model loads.
Everything else (prompts, video windows, QIC_SHARD/QIC_SKIP) is unchanged.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qic_shard  # noqa: E402  (patches the baseline's dataset class)

baseline = qic_shard.baseline
_load = baseline.load_model_and_processor


def _greedy_load(*args, **kwargs):
    model, processor = _load(*args, **kwargs)
    g = model.generation_config
    g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    print(f"qic_greedy: {g}", flush=True)
    return model, processor


baseline.load_model_and_processor = _greedy_load

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
