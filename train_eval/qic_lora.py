"""Official turn-based baseline (greedy, via qic_greedy/qic_shard) with a trained LoRA adapter merged in.
QIC_LORA=<adapter dir> (from sia_train.py). Prompts, windows and decoding are unchanged; combine with qic_online
flags by importing that module instead when needed."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if "--qic_lora" in sys.argv:  # flag form (queue EXTRA) of QIC_LORA
    k = sys.argv.index("--qic_lora"); os.environ["QIC_LORA"] = sys.argv[k + 1]; del sys.argv[k:k + 2]

import qic_greedy  # noqa: E402,F401

baseline = qic_greedy.baseline
_load = baseline.load_model_and_processor


def _load_lora(*args, **kwargs):
    from peft import PeftModel
    model, processor = _load(*args, **kwargs)
    path = os.environ["QIC_LORA"]
    model = PeftModel.from_pretrained(model, path).merge_and_unload().eval()
    print(f"qic_lora: merged {path}", flush=True)
    return model, processor


baseline.load_model_and_processor = _load_lora

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
