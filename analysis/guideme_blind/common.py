"""GuideMe video-blind frontier: shared data / protocol / scoring helpers.

Data   : data/hf_benchmark/annotations.jsonl (alisn/GuideMe_Benchmark @ 2e28648, 473 test videos, annotations only).
Scoring: official/score_stream.py = byte-identical copy of ref_sys/GuideMe/score_stream.py
         (sha256 80b3da20...). We call its own functions:
           evaluate_sample(sample, llm_judger=None, pc_time_window=2.0)   -> CS / NR / FA / PC categories (Stage 3)
           find_match_by_segments + compute_precision_recall_f1           -> sP / sR / sF1 per video (Stage 2)
         The LLM judge (Score, Score_m) is NOT run (no credentials, paid API); see report.
Protocol (paper Sec. 3.4/3.5, default "Anchor-Based" + GT dialogue history; not in the released code, re-implemented):
  anchors  = every GT assistant timestamp;
  silent   = integer seconds strictly between first and last anchor, >= 5 s from every anchor,
             sampled without replacement, n = min(#candidates, #anchors) per video (seeded).
  At query time t the model sees the user request and all GT assistant turns with timestamp < t.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "data" / "hf_cache"))
sys.path.insert(0, str(HERE / "official"))
import score_stream as ss  # noqa: E402  (official, unmodified)

ANNOT = HERE / "data" / "hf_benchmark" / "annotations.jsonl"
SILENT_GAP = 5
NFOLD = 5
DOMS = ["captioncook4d", "egoper", "holoassist", "fitness"]


def load_videos() -> list[dict]:
    out = []
    for line in open(ANNOT, encoding="utf-8"):
        r = json.loads(line)
        conv = r["conversations"]
        req = [c for c in conv if c["from"] == "user"][0]
        turns = sorted(((int(c["timestamp"]), c["value"]) for c in conv if c["from"] == "assistant"), key=lambda x: x[0])
        dom = r["videos"][0].split("/")[0]
        vid = r["video_id"]
        fold = int(hashlib.md5(vid.encode()).hexdigest(), 16) % NFOLD
        out.append(dict(vid=vid, dom=dom, act=r.get("activity_name"), req=req["value"], req_t=int(req["timestamp"]),
                        ts=[t for t, _ in turns], txt=[x for _, x in turns], fold=fold))
    return out


def instances(videos: list[dict], seed: int, protocol: str = "anchor") -> list[dict]:
    """One dict per query (video, t). Features are what a video-blind model sees under GT history."""
    rng = random.Random(seed)
    out = []
    for v in videos:
        ts = v["ts"]
        anchors = set(ts)
        if protocol == "anchor":
            cand = [t for t in range(ts[0], ts[-1] + 1) if min(abs(t - a) for a in ts) >= SILENT_GAP]
            sil = rng.sample(cand, min(len(cand), len(ts)))
            qs = sorted(set(ts) | set(sil))
        elif protocol == "dense":  # released-code replay: first request .. last turn + U[10,15] s (video-end trim unknown)
            qs = list(range(v["req_t"], ts[-1] + rng.randint(10, 15) + 1))
        else:
            raise ValueError(protocol)
        for t in qs:
            k = sum(1 for a in ts if a < t)
            last = ts[k - 1] if k else v["req_t"]
            out.append(dict(vid=v["vid"], dom=v["dom"], act=v["act"], fold=v["fold"], t=t, k=k, e=t - last,
                            is_anchor=t in anchors, gt=v["txt"][ts.index(t)] if t in anchors else "Silent",
                            last_txt=v["txt"][k - 1] if k else None))
    return out


def hash01(vid: str, t: int, salt: str = "") -> float:
    return int(hashlib.md5(f"{salt}{vid}/{t}".encode()).hexdigest(), 16) / 16 ** 32


# ---------------------------------------------------------------- official scoring
def to_samples(inst: list[dict], preds: list[str], videos: list[dict]) -> list[dict]:
    """Merged per-video conversations in the evaluator's format (value = GT or 'Silent', pred = rule output)."""
    req = {v["vid"]: v for v in videos}
    by = {}
    for x, p in zip(inst, preds):
        by.setdefault(x["vid"], []).append({"from": "assistant", "value": x["gt"], "pred": p, "timestamp": x["t"]})
    return [{"video_id": vid, "type": req[vid]["dom"],
             "conversations": [{"from": "user", "value": req[vid]["req"], "timestamp": req[vid]["req_t"]}] + fr}
            for vid, fr in by.items()]


def behavior(samples: list[dict]) -> dict:
    """Official Stage-3 categorisation (judge disabled: its 'Error (no LLM)' label is the PartlyCorrect branch)."""
    ss.PLACEHOLDERS = ss.PLACEHOLDERS_V2
    cnt = {"CS": 0, "NR": 0, "FA": 0, "PC": 0}
    m = {"CorrectSilent": "CS", "NoResponse": "NR", "FalseAlarm": "FA", "Error (no LLM)": "PC", "PartlyCorrect": "PC"}
    for s in samples:
        for r in ss.evaluate_sample(s, None, pc_time_window=2.0):
            cnt[m[r["category"]]] += 1
    n = sum(cnt.values())
    out = {k: 100 * c / n for k, c in cnt.items()}
    out["n"] = n
    out["TNR"] = 100 * cnt["CS"] / max(1, cnt["CS"] + cnt["FA"])
    out["TPR"] = 100 * cnt["PC"] / max(1, cnt["PC"] + cnt["NR"])
    return out


class CachedSTS:
    """Drop-in for the SentenceTransformer passed to the official matcher: same model, embeddings memoised per string
    (all-mpnet-base-v2, CPU). Only .encode is used by ss.get_semantic_match_cost."""

    def __init__(self, cache_file: Path):
        import torch
        import sentence_transformers as sbert
        self.torch = torch
        self.model = sbert.SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        self.file = cache_file
        self.emb: dict[str, np.ndarray] = {}
        if cache_file.exists():
            z = np.load(cache_file, allow_pickle=True)
            self.emb = dict(zip(z["keys"].tolist(), z["vals"]))

    def warm(self, strings):
        new = sorted({s for s in strings if s not in self.emb})
        if new:
            vals = self.model.encode(new, batch_size=64, show_progress_bar=True)
            self.emb.update(zip(new, vals))
            np.savez(self.file, keys=np.array(list(self.emb), dtype=object), vals=np.stack(list(self.emb.values())))

    def encode(self, strings, convert_to_tensor=True, **kw):
        self.warm(strings)
        return self.torch.tensor(np.stack([self.emb[s] for s in strings]))


def alignment(samples: list[dict], sts) -> dict:
    """Official Stage-2 loop (run_metric_match body): per-video segmented matching, metrics averaged over videos."""
    sil = {"silent", "<|silent|>"}
    per = []
    for s in samples:
        outs = []
        for c in s["conversations"]:
            if "pred" not in c:
                continue
            gen = "" if c["pred"].lower().strip() in sil else c["pred"]
            ref = "" if c["value"].lower().strip() in sil else c["value"]
            outs.append(ss.FrameOutput(gen=gen, ref=ref, frame_idx_in_stream=c["timestamp"],
                                       timestamp_in_stream=float(c["timestamp"])))
        mr, _ = ss.find_match_by_segments(outs, sts_model=sts, gaussian_sigma=0.01, batch_size=64, debug=False)
        per.append(ss.compute_precision_recall_f1(mr, semantic_score_threshold=0.5, fps=1))
    return {"sP": 100 * float(np.mean([m["soft_precision"] for m in per])),
            "sR": 100 * float(np.mean([m["soft_recall"] for m in per])),
            "sF1": 100 * float(np.mean([m["soft_f1"] for m in per]))}
