"""Official turn-based baseline (via qic_shard) with optional online alert policies, for real official-protocol runs.

QIC_MEM=1   Streaming Interventions (arXiv 2606.09547) appendix D: the mistake prompt additionally says "Do not
            repeat previously detected mistakes. Here are the feedbacks corresponding to previously detected
            mistakes: [...]" listing this step's feedbacks from the last 60 s (12 checks of 5 s). Inserted right
            before "Assume the recipe step is still in progress." as in the paper's printed prompt.
QIC_GATE=K  emit a mistake feedback only on the K-th consecutive check whose answer is YES, once per run of YES
            answers (other YES answers are returned to the baseline as "NO", so it records nothing).
QIC_STRICT=1  an answer counts as YES only if it starts with yes (after stripping <, quotes, spaces); otherwise the
            baseline's own `"yes" in text` applies. Applies to both prompts.
The step clock is the number of completion checks since the instruction text last changed (the baseline asks the
completion question exactly once per 5 s check). Greedy decoding as in qic_greedy. Nothing else changes.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qic_greedy  # noqa: E402,F401  (greedy decoding + qic_shard dataset patch)

baseline = qic_greedy.baseline
for flag, env in (("--qic_mem", "QIC_MEM"), ("--qic_strict", "QIC_STRICT"), ("--qic_mem_keep", "QIC_MEM_KEEP")):  # flags also accepted (queue EXTRA)
    if flag in sys.argv:
        sys.argv.remove(flag); os.environ[env] = "1"
if "--qic_gate" in sys.argv:
    k = sys.argv.index("--qic_gate"); os.environ["QIC_GATE"] = sys.argv[k + 1]; del sys.argv[k:k + 2]
MEM = os.environ.get("QIC_MEM") == "1"
GATE = int(os.environ.get("QIC_GATE", "1"))
STRICT = os.environ.get("QIC_STRICT") == "1"
# QIC_MEM_KEEP=1: memory = all feedbacks of the last 60 s of video checks, NOT cleared at a new step (Streaming
# Interventions: "recently generated feedbacks (of the last 1 minute)"); clock = global check counter of the video.
KEEP = os.environ.get("QIC_MEM_KEEP") == "1"
MARK = "<<PREV_FEEDBACKS>>"
ANCHOR = "Assume the recipe step is still in progress."

if MEM:
    cfg = baseline.PROMPT_CONFIGS["mistake_inference"]
    assert ANCHOR in cfg.user_text_template
    cfg.user_text_template = cfg.user_text_template.replace(
        ANCHOR, "Do not repeat previously detected mistakes. Here are the feedbacks corresponding to previously "
                f"detected mistakes:{MARK}. " + ANCHOR, 1)

state = {"instr": None, "tick": 0, "mem": [], "run": 0, "fired": False, "gtick": 0, "video": None}
_build, _ask = baseline.build_messages, baseline.get_qwen_vl_output


def _is_yes(text):
    t = text.lower()
    return bool(re.match(r"[\s<\"'*]*yes", t)) if STRICT else "yes" in t


def _build_messages(mode, **kw):
    msgs = _build(mode, **kw)
    if MEM and mode == "mistake_inference":
        now = state["gtick"] if KEEP else state["tick"]
        recent = [f for t, f in state["mem"] if now - t < 12]
        prev = " " + " ".join(f"[{f}]" for f in recent) if recent else " None"
        for part in msgs[1]["content"]:
            if part.get("type") == "text":
                part["text"] = part["text"].replace(MARK, prev)
    return msgs


def _get_output(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    if mode == "instruction_end":
        vid = id(video.base) if hasattr(video, "base") and video.base is not None else None
        if KEEP and vid != state["video"]:
            state.update(video=vid, gtick=0, mem=[])          # new video: memory never crosses videos
        elif KEEP:
            state["gtick"] += 1
        if instruction != state["instr"]:
            state.update(instr=instruction, tick=0, run=0, fired=False, **({} if KEEP else {"mem": []}))
        else:
            state["tick"] += 1
    out = _ask(mode, model, processor, video=video, instruction=instruction, max_new_tokens=max_new_tokens)
    yes = _is_yes(out)
    if mode == "instruction_end":
        return out if yes or not STRICT else ("NO" if "yes" in out.lower() else out)
    if not yes:
        state["run"], state["fired"] = 0, False
        return "NO" if "yes" in out.lower() else out
    state["run"] += 1
    if state["run"] < GATE or state["fired"]:
        return "NO"
    state["fired"] = True
    fb = out.lower().replace("yes.", "").replace("yes", "").strip()
    state["mem"].append((state["gtick"] if KEEP else state["tick"], fb))
    return out


baseline.build_messages = _build_messages
baseline.get_qwen_vl_output = _get_output
print(f"qic_online: MEM={MEM} GATE={GATE} STRICT={STRICT}", flush=True)

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
