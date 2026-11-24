"""Step-aligned frame swap (sufficiency control; inference only).
The official per-step protocol runs on the ORIGINAL video's GT steps and instructions, but for every check of step i the window
is filled with frames of ANOTHER recording of the SAME recipe (swap map, e.g. swap_test36.json), aligned at the START of the
donor's step with the same instruction: window frame k (absolute index in the original) -> donor frame dstart_j + (k - start_i),
continuing naturally into the donor video past its step end. If the model reads the step's progress from the video, its
confirmation time should follow the donor step's duration, not the original's. Steps whose instruction has no counterpart in the
donor keep their own frames (logged as 'unmatched' and excluded from the analysis). Everything else (prompts, decoding, window,
check cadence, scoring) is the official turn-based baseline, via qic_lora (LoRA) or qic_greedy.
    QIC_SWAP_ALIGN=swap_test36.json QIC_LORA=<adapter> QIC_SHARD=i/n python qic_swapalign.py <baseline args>
Writes per-check mapping to <save_root>/<save_file>.align.jsonl."""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
if os.environ.get("QIC_LORA"):
    import qic_lora as _w  # noqa: E402,F401  (also installs qic_greedy + qic_shard)
else:
    import qic_greedy as _w  # noqa: E402,F401
import qic_shard  # noqa: E402
from utils import load_frames_into_array  # noqa: E402
baseline = qic_shard.baseline
SW = json.load(open(os.environ["QIC_SWAP_ALIGN"]))
norm = lambda s: " ".join(s.strip().lower().rstrip(".").split())
CUR = {}
_Base = baseline.QualcommInteractiveCookingDatasetVideos      # qic_shard._Shard
_Orig = _Base.__bases__[0]                                     # unfiltered official dataset class


class _Align(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        full = _Orig(*args, **kwargs)
        self._by_id = {a["video_id"]: a for a in full.annotations}

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        v = d["video_id"]; donor = SW.get(v); CUR.clear(); CUR.update(video=v, donor=donor, arr=None, map={})
        if not donor:
            return d
        da = self._by_id[donor]
        dsteps = [(norm(t[0]), ts[0], max(ts)) for t, ts in zip(da["instruction_segment_texts"], da["instruction_segment_texts_timestamps"])]
        used = set()
        for i, (t, ts) in enumerate(zip(d["gt_texts"], d["gt_text_timestamps"])):
            key = norm(t[0]); j = next((j for j, s in enumerate(dsteps) if s[0] == key and j not in used), None)
            if j is None:
                continue
            used.add(j)
            CUR["map"].setdefault(t[0], []).append(dict(own_step=i, own_start=ts[0], own_end=max(ts), donor_step=j,
                                                        donor_start=dsteps[j][1], donor_end=dsteps[j][2]))
        CUR["donor_paths"] = da["video_frame_paths"]
        print(f"qic_swapalign: {v} <- {donor}: {sum(len(x) for x in CUR['map'].values())} of {len(d['gt_texts'])} steps matched", flush=True)
        return d


baseline.QualcommInteractiveCookingDatasetVideos = _Align
_ask = baseline.get_qwen_vl_output
LOG = None


def _get(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    global LOG
    base = video.base if video.base is not None else video
    s_abs = (video.__array_interface__["data"][0] - base.__array_interface__["data"][0]) // base.strides[0]; e_abs = s_abs + len(video)
    rec = dict(video=CUR.get("video"), donor=CUR.get("donor"), mode=mode, instruction=instruction, start=int(s_abs), end=int(e_abs), matched=False)
    cands = [m for m in CUR.get("map", {}).get(instruction, []) if int(m["own_start"] * 2) <= s_abs]
    if cands:
        m = max(cands, key=lambda m: m["own_start"])
        if CUR["arr"] is None:
            CUR["arr"] = load_frames_into_array(CUR["donor_paths"], video_input_resolution=(video.shape[2], video.shape[1]))
        o0, d0 = int(m["own_start"] * 2), int(m["donor_start"] * 2); n = len(CUR["arr"])
        idx = [min(max(d0 + (k - o0), 0), n - 1) for k in range(s_abs, e_abs)]
        video = CUR["arr"][idx]
        rec.update(matched=True, own_step=m["own_step"], donor_step=m["donor_step"], donor_first=idx[0], donor_last=idx[-1])
    out = _ask(mode, model, processor, video=video, instruction=instruction, max_new_tokens=max_new_tokens)
    rec["out"] = out[:80]
    if LOG is None:
        LOG = open(os.environ.get("QIC_ALIGN_LOG", "align.jsonl"), "a")
    LOG.write(json.dumps(rec) + "\n"); LOG.flush()
    return out


baseline.get_qwen_vl_output = _get
print("qic_swapalign: active", flush=True)
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
