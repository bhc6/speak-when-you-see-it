"""Appendix control: FIXED-LENGTH observation window (removes the window-length = elapsed-time cue).
The official per-step protocol is unchanged (GT restarts, 5-s checks, prompts, parsing, greedy decoding); only the clip
handed to the model is replaced: instead of [step GT start, t] (capped at 240 frames), the model sees the last
QIC_FIXWIN frames of the video ending at t, crossing step boundaries (shorter only in the first QIC_FIXWIN frames of the
video). The baseline passes a slice of the whole-video array, so its end index is recovered from the view offset.
    QIC_FIXWIN=60 [QIC_LORA=dir] python qic_fixwin.py <baseline args>      (also --fixwin N / --qic_lora DIR)"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "--fixwin" in sys.argv:
    k = sys.argv.index("--fixwin"); os.environ["QIC_FIXWIN"] = sys.argv[k + 1]; del sys.argv[k:k + 2]
if "--qic_lora" in sys.argv or os.environ.get("QIC_LORA"):
    import qic_lora as _w  # noqa: E402  (greedy + LoRA merge)
else:
    import qic_greedy as _w  # noqa: E402
baseline = _w.baseline
L = int(os.environ["QIC_FIXWIN"]); _ask = baseline.get_qwen_vl_output; stats = {"n": 0, "short": 0}

def _fixed(video):
    base = video.base
    assert base is not None and base.ndim == video.ndim and base.strides == video.strides, "clip is not a view of the video"
    off = (video.__array_interface__["data"][0] - base.__array_interface__["data"][0]) // base.strides[0]
    end = off + len(video); clip = base[max(end - L, 0):end]
    stats["n"] += 1; stats["short"] += len(clip) < L
    if stats["n"] % 500 == 1: print(f"qic_fixwin: check {stats['n']} end={end} len {len(clip)} (orig {len(video)}), short so far {stats['short']}", flush=True)
    return clip

def _get(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    return _ask(mode, model, processor, video=_fixed(video), instruction=instruction, max_new_tokens=max_new_tokens)

baseline.get_qwen_vl_output = _get
print(f"qic_fixwin: fixed window {L} frames", flush=True)
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
