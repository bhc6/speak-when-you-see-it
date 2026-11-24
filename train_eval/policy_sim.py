"""Offline alert-policy simulator on dense scores (qic_dense.py output), scored with the official eval's own counting.

Bounded turn-based protocol: each GT step is played from its GT start over the densely scored checks (up to GT end +
margin); at every check the completion rule is asked first (fires -> "Success" at that time, step ends, no mistake
question at that check, exactly as the baseline), otherwise the mistake rule may emit a "Feedback". A step whose
completion never fires ends at the last scored check (the official protocol would keep going to the video end and
drop all later steps; that knock-on is not simulated -- report it).

Rules (p = P(yes) / (P(yes) + P(no)) from the first generated token; "text" uses the greedy text like the baseline):
  text                    fire when "yes" in the greedy text (baseline decision rule)
  thr:T                   fire when p >= T at a check
  k:K:T                   fire on the K-th consecutive check with p >= T (once per run of such checks)
  cusum:D:H               S <- max(0, S + logit(p) - D); fire when S >= H, then S <- 0
  never                   never fire
plus mistake option cap:M (at most M alerts per step).

Scoring imports eval.py from the official repo (cwd must be the eval repo, PYTHONPATH=./) and runs its run() with
BERTScore stubbed to 0 for speed (fast=True); fast=False keeps the official BERT/ROUGE. Counts are parsed from its
printout, so they are the official numbers by construction.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re

EPS = 1e-6


def norm_p(pair):
    y, n = pair
    return y / (y + n) if y + n > 0 else 0.5


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


class Rule:
    def __init__(self, spec: str, which: str):
        self.spec, self.which = spec, which
        parts = spec.split(":")
        self.kind, self.args = parts[0], [float(x) for x in parts[1:]]
        self.reset()

    def reset(self):
        self.run, self.S, self.fired_run = 0, 0.0, False

    def step(self, c) -> bool:
        k = self.kind
        if k == "never":
            return False
        if k == "text":
            return "yes" in c[f"{self.which}_text"].lower()
        p = norm_p(c[f"p_{self.which}"])
        if k == "thr":
            return p >= self.args[0]
        if k == "k":
            K, T = int(self.args[0]), self.args[1]
            if p >= T:
                self.run += 1
                if self.run >= K and not self.fired_run:
                    self.fired_run = True
                    return True
            else:
                self.run, self.fired_run = 0, False
            return False
        if k == "cusum":
            D, H = self.args
            self.S = max(0.0, self.S + logit(p) - D)
            if self.S >= H:
                self.S = 0.0
                return True
            return False
        raise ValueError(self.spec)


def feedback_text(t: str) -> str:  # same cleaning as the baseline
    return "Feedback: " + t.lower().replace("yes.", "").replace("yes", "").strip()


def simulate(dense, done_spec: str, mis_spec: str, cap: int = 10**9):
    preds = []
    for v in dense:
        texts, stamps = [], []
        for s in v["steps"]:
            texts.append("Instruction: " + s["instruction"]); stamps.append(s["gt_start"])
            dr, mr, n_alert = Rule(done_spec, "done"), Rule(mis_spec, "mis"), 0
            for c in s["checks"]:
                if dr.step(c):
                    texts.append("Success: Good job!"); stamps.append(c["t"]); break
                if mr.step(c) and n_alert < cap:
                    texts.append(feedback_text(c["mis_text"])); stamps.append(c["t"]); n_alert += 1
        preds.append({"video_id": v["video_id"], "pred_texts": texts, "pred_timestamps": stamps})
    return preds


_EVAL = {}


def score(preds, plan_set="advanced_planning", split="validation", fast=True):
    if "mod" not in _EVAL:
        if fast:  # no BERTScorer at all (it would load roberta-large on the GPU in every worker)
            import sys, types
            stub = types.ModuleType("bert_score")
            stub.BERTScorer = lambda *a, **k: None
            sys.modules.setdefault("bert_score", stub)
        import eval as ev  # official repo, cwd/PYTHONPATH must point at it
        import utils
        _EVAL["mod"], _EVAL["bert"] = ev, ev.get_bert_score
        _EVAL["ds"] = {}
    ev = _EVAL["mod"]
    key = (plan_set, split)
    if key not in _EVAL["ds"]:
        _EVAL["ds"][key] = ev.load_dataset("neurips2025", plan_set, split)
    ev.get_bert_score = (lambda a, b: 0.0) if fast else _EVAL["bert"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.run(_EVAL["ds"][key], preds)
    blocks = buf.getvalue().split("<<<")
    last = blocks[-1]
    ic = float(re.search(r"IC-Acc: ([\d.]+)", last).group(1))
    tp, fp, tn, fn = map(int, re.search(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", last).groups())
    bert, rouge = map(float, re.search(r"BERT Score: ([\d.na]+), Mistake Mean Rouge-L Score: ([\d.na]+)", last).groups())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"IC": ic, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "P": p, "R": r, "F1": f1, "BERT": bert, "ROUGE": rouge}


def load_dense(path):
    return json.load(open(path))


def ic_denominators(dense):
    """Per-video IC denominator = number of GT steps without mistakes (eval.py's total_instructions_without_mistakes)."""
    return {v["video_id"]: sum(not s["has_mistake"] for s in v["steps"]) for v in dense}


def score_pervideo(preds, plan_set="advanced_planning", split="validation"):
    """Per-video official counts in one eval pass. Requires the cwd/PYTHONPATH eval module to be analysis/evalpv
    (official eval + one print line "PV vid ic den tp fp tn fn" with cumulative counts after each video).
    Returns {video_id: [ic, den, tp, fp, tn, fn]}."""
    score(preds, plan_set, split, fast=True)  # loads module/dataset, warms up
    ev = _EVAL["mod"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.run(_EVAL["ds"][(plan_set, split)], preds)
    prev, out = [0] * 6, {}
    for line in buf.getvalue().splitlines():
        if line.startswith("PV "):
            f = line.split(); cur = list(map(int, f[2:8]))
            out[f[1]] = [c - q for c, q in zip(cur, prev)]; prev = cur
    assert len(out) == len(preds), (len(out), len(preds))
    return out
