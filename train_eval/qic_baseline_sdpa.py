"""Run the official Qwen3-VL baseline (ref_sys/ai_coach_cooking_2026/qwen3_vl_baseline.py) unchanged, except that the
model loads with PyTorch's sdpa attention instead of flash_attention_2 (flash-attn has no Windows build and needs
Ampere or newer).

QIC_DTYPE=fp16 also runs it in float16 instead of the baseline's bfloat16: Volta (V100) has no bf16 units, so bf16
is emulated there (measured ~6 min per 240-frame call vs ~8 s in fp16). The baseline names torch.bfloat16 in three
places (model load, input cast, autocast); its module-level `torch` is swapped for a proxy whose bfloat16 is float16,
so all three change together and nothing else does.

The cloned repo is not edited: this wrapper patches the baseline's module before calling its own run(parse_args()).
All command-line arguments are the baseline's own, e.g.

    cd ref_sys/ai_coach_cooking_2026
    QIC_DTYPE=fp16 python ../../experiments/plan_repr/qic_baseline_sdpa.py \
        --captaincook4d_root <root>/captain_cook_4d/gopro --plan_set advanced_planning --split validation \
        --model_id Qwen/Qwen3-VL-2B-Instruct --turn_based
"""

from __future__ import annotations

import functools
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_sys", "ai_coach_cooking_2026")
sys.path.insert(0, os.path.abspath(REPO))

import qwen3_vl_baseline as baseline  # noqa: E402

_orig = baseline.Qwen3VLForConditionalGeneration.from_pretrained


@functools.wraps(_orig)
def _from_pretrained_sdpa(*args, **kwargs):
    kwargs["attn_implementation"] = "sdpa"
    return _orig(*args, **kwargs)


baseline.Qwen3VLForConditionalGeneration.from_pretrained = _from_pretrained_sdpa


class _Fp16Torch:
    """`torch` as seen by the baseline module only: identical except that bfloat16 means float16."""

    bfloat16 = torch.float16

    def __getattr__(self, name):
        return getattr(torch, name)


DTYPE = os.environ.get("QIC_DTYPE", "bf16")
if DTYPE == "fp16":
    baseline.torch = _Fp16Torch()
elif DTYPE != "bf16":
    sys.exit(f"QIC_DTYPE must be bf16 or fp16, not {DTYPE!r}")

if __name__ == "__main__":
    print(f"qic_baseline_sdpa: attention sdpa, dtype {DTYPE}", flush=True)
    baseline.run(baseline.parse_args())
