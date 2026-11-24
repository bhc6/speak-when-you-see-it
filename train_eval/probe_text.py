"""Phase A, CPU part: text-only encodings with the same frozen Qwen3-VL-2B (no GPU; CUDA_VISIBLE_DEVICES="").
1. Candidate interventions: every distinct mistake-feedback text in QICD advanced_planning train + validation,
   a success message and a silence message, embedded as 'The cooking assistant says to the person: "<text>"'.
2. Text-only control situations: every item of the probe sampler (same steps/times), same prompt as qic_probe.py
   but without the video -> isolates what the video adds.
Same layers and last-prompt-token convention as qic_probe.py. Output: results/probe/text_encodings.npz + .json"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import qic_probe as P

if os.environ.get("PROBE_LAYERS"): P.LAYERS = [int(x) for x in os.environ["PROBE_LAYERS"].split(",")]
torch.set_num_threads(int(os.environ.get("NT", "48")))
mid = os.environ.get("PROBE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
OUT = os.environ.get("PROBE_OUT", "WORKSPACE/qic/results/probe"); os.makedirs(OUT, exist_ok=True)
if "Qwen3.5" in mid:
    from transformers import AutoModelForImageTextToText as _M
else:
    _M = Qwen3VLForConditionalGeneration
model = _M.from_pretrained(mid, dtype=torch.bfloat16, device_map="cpu").eval()
proc = AutoProcessor.from_pretrained(mid)
SUCCESS, SILENCE = "Good job! You have completed this step.", "(The assistant stays silent and says nothing.)"


@torch.no_grad()
def enc(msgs):
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = proc.tokenizer([text], return_tensors="pt")
    out = model.model.language_model(**ids, output_hidden_states=True, use_cache=False)
    return np.stack([out.hidden_states[L][0, -1].float().numpy() for L in P.LAYERS]).astype(np.float16)


cands = []
for sp in ("train", "validation"):
    for r in load_dataset("qualcomm/qualcomm-interactive-cooking-dataset", "advanced_planning", split=sp):
        for t, y in zip(r["output_texts"], r["output_types"]):
            if "mistake" in y: cands.append((t, sp, y, r["video_id"], r["activity_name"]))
uniq = {}
for c in cands: uniq.setdefault(c[0], c)
cands = [{"text": t, "split": c[1], "gt_type": c[2], "video_id": c[3], "activity": c[4], "kind": "mistake"} for t, c in uniq.items()]
cands += [{"text": SUCCESS, "kind": "success"}, {"text": SILENCE, "kind": "silence"}]
print("candidates", len(cands), flush=True)
SKIP_CANDS = os.path.exists(f"{OUT}/text_candidates.npz")
cand_h = [] if not SKIP_CANDS else None
for i, c in enumerate([] if SKIP_CANDS else cands):
    cand_h.append(enc([{"role": "user", "content": [{"type": "text", "text": f'The cooking assistant says to the person: "{c["text"]}"'}]}]))
    if i % 100 == 0: print("cand", i, flush=True)
if not SKIP_CANDS: np.savez(f"{OUT}/text_candidates.npz", h=np.stack(cand_h))
json.dump(cands, open(f"{OUT}/text_candidates.json", "w"))

# text-only situations: replicate the sampler on GT without frames
import importlib.util
_spec = importlib.util.spec_from_file_location("eval_utils", "WORKSPACE/qic/ref_sys/qualcomm_interactive_cooking_eval/utils.py")
sys.path.insert(0, "WORKSPACE/qic/ref_sys/qualcomm_interactive_cooking_eval")
import types; _bs = types.ModuleType("bert_score"); _bs.BERTScorer = lambda *a, **k: None; sys.modules["bert_score"] = _bs
sys.modules.pop("data", None)  # eval repo's data.py, not the quick-start one
_eu = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_eu)
ds = _eu.load_dataset("neurips2025", "advanced_planning", "validation")
items, H = [], []
for i in range(len(ds)):
    g = ds[i]
    d = {"num_of_instruction_segments": len(g["texts"]), "gt_texts": g["texts"], "gt_text_timestamps": g["texts_timestamps"],
         "gt_text_types": g["texts_types"], "gt_has_mistake": g["has_mistake"]}
    for it in P.sample_items(d):
        instr = g["texts"][it["step"]][0]
        H.append(enc([{"role": "system", "content": P.SYSTEM},
                      {"role": "user", "content": [{"type": "text", "text": P.QUESTION.format(instruction=instr)}]}]))
        it.update(video_id=g["video_id"], instruction=instr); items.append(it)
    print("textsit", g["video_id"], len(items), flush=True)
np.savez(f"{OUT}/text_situations.npz", h=np.stack(H)); json.dump(items, open(f"{OUT}/text_situations.json", "w"))
print("done", flush=True)
