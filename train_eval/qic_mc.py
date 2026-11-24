"""Run any qic_* wrapper on Ego-MC-Bench instead of QICD: replaces the baseline's dataset class with one built from
the official eval repo's EgoMCBench annotations (same segmentation the eval scores with), frames from
mc_prepare.py. Usage: QIC_MC_ROOT=<root> QIC_MC_WRAP=qic_greedy python qic_mc.py <baseline args>
(--captaincook4d_root/--plan_set/--split are required by the parser but ignored)."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EVAL = os.environ.get("QIC_EVAL_DIR", os.path.join(HERE, "..", "..", "ref_sys", "qualcomm_interactive_cooking_eval"))


def _eval_mod(name):
    spec = importlib.util.spec_from_file_location(f"qic_eval_{name}", os.path.join(EVAL, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


class MCVideos:
    def __init__(self, *args, model_fps=2, **kwargs):
        self.model_fps = model_fps
        self.ds = _eval_mod("data").EgoMCBench(split="test")
        root = os.environ["QIC_MC_ROOT"]
        self.items = []
        for a in self.ds.annotations:
            d = f"{root}/frames_2fps/{a['video_id']}"
            fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x[6:-4]))
            self.items.append({"video_id": a["video_id"], "video_frame_paths": [f"{d}/{f}" for f in fs],
                               "video_frame_timestamps": [i / model_fps for i in range(len(fs))],
                               "global_start_timestamp": a["instruction_segment_texts_timestamps"][0][0],
                               "num_of_instruction_segments": len(a["instruction_segment_texts"]),
                               "gt_texts": a["instruction_segment_texts"], "gt_text_timestamps": a["instruction_segment_texts_timestamps"],
                               "gt_text_types": a["instruction_segment_texts_types"], "gt_has_mistake": a["instrcution_segment_has_mistake"]})
        self.annotations = self.items  # qic_shard filters self.annotations

    def __len__(self): return len(self.annotations)

    def __getitem__(self, i): return self.annotations[i]


wrap = importlib.import_module(os.environ.get("QIC_MC_WRAP", "qic_greedy"))
base = wrap.baseline
_shard = sys.modules["qic_shard"]._Shard


class MCShard(MCVideos):  # same video selection rule as qic_shard (QIC_SHARD / QIC_SKIP)
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        keep = [x for j, x in enumerate(self.annotations) if j % sys.modules["qic_shard"].n == sys.modules["qic_shard"].i
                and x["video_id"] not in sys.modules["qic_shard"].skip]
        print(f"qic_mc: {len(keep)} of {len(self.annotations)} Ego-MC videos", flush=True); self.annotations = keep


base.QualcommInteractiveCookingDatasetVideos = MCShard

if __name__ == "__main__":
    base.run(base.parse_args())
