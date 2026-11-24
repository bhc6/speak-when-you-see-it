"""qic_shard.py with the model's default sampling and a fixed seed, for repeated runs of the official baseline setting.

Generation is left to the model's generation_config (Qwen3-VL-2B-Instruct: do_sample, temperature 0.7, top_p 0.8,
top_k 20), exactly as the baseline runs it. --qic_seed S (removed before the baseline parses its arguments) reseeds
random/numpy/torch before each video from (S, video_id), so a video's output does not depend on which shard runs it
or in what order.
"""

from __future__ import annotations

import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import qic_shard  # noqa: E402  (patches the baseline's dataset class)
import torch  # noqa: E402

baseline = qic_shard.baseline
seed = 0
if "--qic_seed" in sys.argv:
    k = sys.argv.index("--qic_seed")
    seed = int(sys.argv[k + 1])
    del sys.argv[k:k + 2]

_cls = baseline.QualcommInteractiveCookingDatasetVideos
_getitem = _cls.__getitem__


def _seeded_getitem(self, idx):
    item = _getitem(self, idx)
    s = zlib.crc32(f"{seed}:{item['video_id']}".encode())
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    print(f"qic_sampled: seed {seed} video {item['video_id']} -> {s}", flush=True)
    return item


_cls.__getitem__ = _seeded_getitem
_load = baseline.load_model_and_processor


def _load_logged(*args, **kwargs):
    model, processor = _load(*args, **kwargs)
    print(f"qic_sampled: {model.generation_config}", flush=True)
    return model, processor


baseline.load_model_and_processor = _load_logged

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
