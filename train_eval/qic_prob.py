"""Official turn-based baseline with probability-threshold decisions (greedy decoding, optional LoRA).

The two official questions are asked exactly as in qwen3_vl_baseline.get_qwen_vl_output (same prompts, windows,
preprocessing). Instead of `"yes" in generated_text`, the decision uses the first-token probability
P(yes) / (P(yes) + P(no)):
  completion prompt : p_done >= QIC_TAU_DONE  -> returns "YES." else "NO."
  mistake prompt    : p_mis  >= QIC_TAU_MIS   -> returns "YES. <feedback>", where the feedback is generated
                      greedily with the assistant turn prefixed by "YES." (so it is the model's own correction text);
                      else "NO."
Every decision (p, threshold, decision) is appended to QIC_PROB_LOG (jsonl) for auditing.
Flags (queue EXTRA form): --qic_lora DIR --tau_done X --tau_mis Y --prob_log FILE.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for flag, env in (("--qic_lora", "QIC_LORA"), ("--tau_done", "QIC_TAU_DONE"), ("--tau_mis", "QIC_TAU_MIS"), ("--prob_log", "QIC_PROB_LOG"),
                  ("--order", "QIC_ORDER")):
    if flag in sys.argv:
        k = sys.argv.index(flag); os.environ[env] = sys.argv[k + 1]; del sys.argv[k:k + 2]

import torch  # noqa: E402

if os.environ.get("QIC_LORA"):
    import qic_lora as _w  # noqa: E402  (greedy + LoRA merge)
else:
    import qic_greedy as _w  # noqa: E402
baseline = _w.baseline
TAU_D = float(os.environ.get("QIC_TAU_DONE", "0.5")); TAU_M = float(os.environ.get("QIC_TAU_MIS", "0.5"))
LOG = os.environ.get("QIC_PROB_LOG")
ORDER = os.environ.get("QIC_ORDER", "done_first")   # done_first = baseline order; mistake_first = joint decision
_ids = {}
_pending = {"fb": None}


def _yes_no(tok):
    if not _ids:
        for i in range(len(tok)):
            s = tok.decode([i]).strip().lower().strip(".,:;!'\"<>()[]")
            if s == "yes": _ids.setdefault("y", []).append(i)
            elif s == "no": _ids.setdefault("n", []).append(i)
    return _ids["y"], _ids["n"]


@torch.no_grad()
def _score_and_answer(mode, model, processor, video, instruction, max_new_tokens):
    y_ids, n_ids = _yes_no(processor.tokenizer)
    text = processor.apply_chat_template(baseline.build_messages(mode, instruction=instruction), tokenize=False, add_generation_prompt=True)
    meta = [{"fps": 2, "total_num_frames": len(video)}]
    inp = processor(text=[text], images=None, videos=video, padding=True, video_metadata=meta, return_tensors="pt").to("cuda").to(torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p = torch.softmax(model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
    py, pn = float(p[y_ids].sum()), float(p[n_ids].sum()); q = py / max(py + pn, 1e-9)
    tau = TAU_D if mode == "instruction_end" else TAU_M
    yes = q >= tau
    out = "NO."
    if yes and mode == "instruction_end":
        out = "YES."
    elif yes:
        inp2 = processor(text=[text + "YES."], images=None, videos=video, padding=True, video_metadata=meta, return_tensors="pt").to("cuda").to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            g = model.generate(**inp2, max_new_tokens=max_new_tokens, do_sample=False)
        fb = processor.batch_decode([g[0][inp2.input_ids.shape[1]:]], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        out = "YES. " + (fb or "You made a mistake.")
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps({"mode": mode, "instr": instruction[:80], "n": len(video), "p": round(q, 4), "tau": tau, "yes": yes}) + "\n")
    return out


def get_output(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    """mistake_first: at every check the mistake question is scored first; if it fires, the completion question
    returns NO (step continues) and the following mistake call returns the stored feedback. Otherwise completion is
    decided by its own threshold and the mistake call (if any) returns NO. The baseline's call sequence is unchanged."""
    if ORDER != "mistake_first":
        return _score_and_answer(mode, model, processor, video, instruction, max_new_tokens)
    if mode == "instruction_end":
        m = _score_and_answer("mistake_inference", model, processor, video, instruction, 1024)
        if m.startswith("YES"):
            _pending["fb"] = m; return "NO."
        _pending["fb"] = None
        return _score_and_answer("instruction_end", model, processor, video, instruction, max_new_tokens)
    fb, _pending["fb"] = _pending["fb"], None
    return fb or "NO."


baseline.get_qwen_vl_output = get_output
print(f"qic_prob: order={ORDER} tau_done={TAU_D} tau_mis={TAU_M} lora={os.environ.get('QIC_LORA')}", flush=True)

if __name__ == "__main__":
    baseline.run(baseline.parse_args())
