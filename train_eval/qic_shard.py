"""Run the official Qwen3-VL baseline (ref_sys/ai_coach_cooking_2026/qwen3_vl_baseline.py) on one shard of the videos,
so several copies can share one GPU. Per-video behaviour is unchanged: the baseline loops over videos independently
and resets all state per video; this wrapper only narrows the dataset it loops over.

QIC_SHARD=i/n keeps videos whose position in the dataset is i mod n. QIC_SKIP=<json> [<json> ...] also drops videos
already present in earlier prediction files, so a stopped run can be resumed. Attention and dtype stay the
baseline's own (flash_attention_2, bfloat16). The cloned repo is not edited.

    QIC_SHARD=0/3 QIC_SKIP=results/a.json python qic_shard.py <baseline args> --save_file shard0.json
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref_sys", "ai_coach_cooking_2026")
sys.path.insert(0, os.path.abspath(REPO))

import qwen3_vl_baseline as baseline  # noqa: E402

i, n = (int(x) for x in os.environ.get("QIC_SHARD", "0/1").split("/"))
skip = {p["video_id"] for f in os.environ.get("QIC_SKIP", "").split() if os.path.exists(f)
        for p in json.load(open(f))}


class _Shard(baseline.QualcommInteractiveCookingDatasetVideos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        keep = [a for k, a in enumerate(self.annotations) if k % n == i and a["video_id"] not in skip]
        print(f"qic_shard {i}/{n}: {len(keep)} of {len(self.annotations)} videos ({len(skip)} skipped as done)",
              flush=True)
        self.annotations = keep

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        if os.environ.get("QIC_SWAP"):
            # visual-dependence control: replace this video's frames by those of ANOTHER validation video of the SAME
            # recipe (fixed derangement), keeping the GT steps / timestamps / instructions of the original video.
            # Length is matched by truncation or by repeating the last frame. Everything else is unchanged.
            sw = json.load(open(os.environ["QIC_SWAP"]))
            other = sw.get(d["video_id"])
            if other:
                src = os.path.join(os.path.dirname(d["video_frame_paths"][0]), "..", other + "_360p")
                fs = sorted((f for f in os.listdir(src) if f.endswith(".jpg") and f.startswith("frame_")),
                            key=lambda x: int(x.split("_")[1].split(".")[0]))
                paths = [os.path.join(src, f) for f in fs]
                n0 = len(d["video_frame_paths"])
                d["video_frame_paths"] = (paths + [paths[-1]] * n0)[:n0]
                print(f"qic_swap: {d['video_id']} <- frames of {other}", flush=True)
        return d


baseline.QualcommInteractiveCookingDatasetVideos = _Shard

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
