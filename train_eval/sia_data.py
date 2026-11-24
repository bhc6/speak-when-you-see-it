"""Training situations for Situation-Intervention Alignment (SIA) from QICD advanced_planning *train* only.

NOTE: the quick-start loader (ai_coach_cooking_2026/data.py) maps split="train" to train+validation; this module
reads the HF split directly so validation never enters training.

Situations per GT step (same windows as the official turn-based baseline: step GT start -> t, <= max_buffer frames):
  mistake   at each GT mistake-feedback time            target = that feedback text
  success   at the GT end of a mistake-free step        target = success
  silence   from start+10 s every `stride` s, >= 15 s from any GT feedback of the step and <= end-15 s
  shifted   the same feedback moved out of its +-15 s window inside the same step (start+5 .. fb-20 s):
            target = silence, and that feedback is a hard negative ("right words, wrong time")
Hard negatives for every situation: mistake feedbacks of the same activity from other videos (other participants);
the success/silence strings are always in the candidate set.
"""
from __future__ import annotations

import collections
import json
import os
import random

from datasets import load_dataset

HF = "qualcomm/qualcomm-interactive-cooking-dataset"
SUCCESS = "Good job! You have completed this step."
SILENCE = "(The assistant stays silent and says nothing.)"


def segments(r):
    """Instruction segments exactly as ai_coach_cooking_2026/data.py preprocess_data."""
    # The official loader means to drop a dangling final instruction, but it rebinds annotation[k] after having
    # captured output_texts etc., so the loop still sees it and it closes the previous segment. Match that.
    texts, ts, types = list(r["output_texts"]), list(r["output_timestamps"]), list(r["output_types"])
    segs, cur = [], None
    for x, t, y in zip(texts, ts, types):
        if y == "instruction" or "finish_all" in y:
            if cur: segs.append(cur)
            cur = {"texts": [x], "ts": [t], "types": [y], "has_mistake": False}
        else:
            cur["texts"].append(x); cur["ts"].append(t); cur["types"].append(y)
            cur["has_mistake"] |= "mistake" in y
    return segs  # like the official loader, the trailing segment (finish_all / unfinished) is not a step


def frame_dir(root, vid):
    for d in (vid + "_360p", vid):
        p = os.path.join(root, d)
        if os.path.isdir(p): return p
    return None


def build(frames_root, split="train", plan_set="advanced_planning", stride=20.0, ids=None, seed=0):
    rows = load_dataset(HF, plan_set, split=split).to_list()
    rows = [r for r in rows if frame_dir(frames_root, r["video_id"]) and (ids is None or r["video_id"] in ids)]
    bank = collections.defaultdict(list)          # activity -> [(video_id, feedback)]
    for r in rows:
        for x, y in zip(r["output_texts"], r["output_types"]):
            if "mistake" in y: bank[r["activity_name"]].append((r["video_id"], x))
    rng = random.Random(seed); ex = []
    for r in rows:
        vid, act = r["video_id"], r["activity_name"]
        other = [x for v, x in bank[act] if v != vid]
        for si, s in enumerate(segments(r)):
            start, end = s["ts"][0], max(s["ts"])
            fb = [(t, x) for x, t, y in zip(s["texts"][1:], s["ts"][1:], s["types"][1:]) if "feedback" in y]
            mis = [(t, x) for x, t, y in zip(s["texts"][1:], s["ts"][1:], s["types"][1:]) if "mistake" in y]
            base = {"video_id": vid, "activity": act, "step": si, "instruction": s["texts"][0], "start": start, "end": end}
            hard = rng.sample(other, min(8, len(other)))
            for t, x in mis:
                ex.append(dict(base, kind="mistake", t=t, target=x, hard=[h for h in hard if h != x]))
            if not s["has_mistake"]:
                ex.append(dict(base, kind="success", t=end, target=SUCCESS, hard=hard))
            t = start + 10
            while t <= end - 15:
                if all(abs(t - f) >= 15 for f, _ in fb):
                    ex.append(dict(base, kind="silence", t=t, target=SILENCE, hard=hard))
                t += stride
            for f, x in mis:
                t = f - 20 - rng.uniform(0, 10)
                if t >= start + 5 and all(abs(t - g) >= 15 for g, _ in fb):
                    ex.append(dict(base, kind="shifted", t=t, target=SILENCE, hard=[x] + hard[:7]))
    return ex, {a: sorted({x for _, x in v}) for a, v in bank.items()}


if __name__ == "__main__":
    import sys
    ex, bank = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train")
    print(len(ex), collections.Counter(e["kind"] for e in ex), "videos", len({e["video_id"] for e in ex}),
          "bank", sum(len(v) for v in bank.values()))


def err_cat(y):
    """'feedback_action_aligned_mistake_technique_error' -> 'technique' (alignment prefix dropped: content level)."""
    return y.split("mistake_")[-1].replace("_error", "")


def build_v2(frames_root, split="train", plan_set="advanced_planning", stride=20.0, ids=None, seed=0):
    """Round-2 situations. Differences from build():
    mistake  at f, f+5, f+10 (inside the eval's +-15 s TP window, matching the 5 s check rate), each with the
             feedback's error category; t >= step start + 5
    success  at end and end+5 (no-mistake steps; inside the IC +-15 s window)
    silence / shifted as before. Bank entries carry (video_id, text, category) for content-level positives."""
    rows = load_dataset(HF, plan_set, split=split).to_list()
    rows = [r for r in rows if frame_dir(frames_root, r["video_id"]) and (ids is None or r["video_id"] in ids)]
    bank = collections.defaultdict(list)          # activity -> [(video_id, feedback, category)]
    for r in rows:
        for x, y in zip(r["output_texts"], r["output_types"]):
            if "mistake" in y: bank[r["activity_name"]].append((r["video_id"], x, err_cat(y)))
    rng = random.Random(seed); ex = []
    for r in rows:
        vid, act = r["video_id"], r["activity_name"]
        other = [(x, c) for v, x, c in bank[act] if v != vid]
        for si, s in enumerate(segments(r)):
            start, end = s["ts"][0], max(s["ts"])
            fb = [(t, x) for x, t, y in zip(s["texts"][1:], s["ts"][1:], s["types"][1:]) if "feedback" in y]
            mis = [(t, x, err_cat(y)) for x, t, y in zip(s["texts"][1:], s["ts"][1:], s["types"][1:]) if "mistake" in y]
            base = {"video_id": vid, "activity": act, "step": si, "instruction": s["texts"][0], "start": start, "end": end}
            hard = rng.sample(other, min(12, len(other)))
            for f, x, c in mis:
                for dt in (0, 5, 10):   # QICD feedback is given just after the mistake: only times it can be visible
                    if f + dt >= start + 5:
                        ex.append(dict(base, kind="mistake", t=f + dt, target=x, cat=c, hard=[h for h in hard if h[0] != x]))
            if not s["has_mistake"]:
                for dt in (0, 5):       # completion must be visible: GT end and +5 s (inside the IC +-15 s window)
                    if end + dt >= start + 5:
                        ex.append(dict(base, kind="success", t=end + dt, target=SUCCESS, cat=None, hard=hard))
            t = start + 10
            while t <= end - 15:
                if all(abs(t - f) >= 15 for f, _ in fb):
                    ex.append(dict(base, kind="silence", t=t, target=SILENCE, cat=None, hard=hard))
                t += stride
            for f, x, c in mis:
                t = f - 20 - rng.uniform(0, 10)
                if t >= start + 5 and all(abs(t - g) >= 15 for g, _ in fb):
                    ex.append(dict(base, kind="shifted", t=t, target=SILENCE, cat=None, hard=[(x, c)] + hard[:11]))
    return ex, {a: sorted({(x, c) for _, x, c in v}) for a, v in bank.items()}


COMIST = "qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes"


def build_v3(frames_root, split="train", plan_set="advanced_planning", stride=20.0, ids=None, seed=0, comist=True, unmatched=False):
    """Round-3 situations = build_v2 plus
    (a) elapsed-matched 'not done yet' negatives (kind 'notdone', target silence): for every success situation at
        elapsed time E, one situation at the same elapsed time E in a random other step of the same video that lasts
        > E + 20 s -> completion can no longer be predicted from elapsed time alone;
    (b) Ego-CoMist CaptainCook4D counterfactual pairs whose video is in this split (train only): the counterfactual
        instruction is the step instruction, window starts at the pair's start_time; mistake situations at the
        feedback time f, f+5, f+10 (target = counterfactual feedback, category = mistake_type); one pre-feedback
        situation at f-20..f-30 (kind 'shifted', target silence, the feedback as hard negative);
    (c) elapsed-matched 'clean' negatives for every mistake situation (kind 'clean', target silence, mistake prompt).
    unmatched=True (control): the same steps, counts, prompts and targets for (a) and (c), but the time inside the chosen
    step is drawn uniformly (notdone: start+5 .. end-20; clean: start+5 .. end-15) instead of start + E, from a separate
    RNG so that every other random choice is identical to the matched build."""
    ex, bank = build_v2(frames_root, split, plan_set, stride, ids, seed)
    rng = random.Random(seed + 7); urng = random.Random(seed + 13)
    steps = collections.defaultdict(list)
    for e in ex: steps[e["video_id"]].append((e["step"], e["start"], e["end"], e["instruction"], e["activity"]))
    for v in steps: steps[v] = sorted(set(steps[v]))
    add = []
    for e in ex:
        if e["kind"] != "success": continue
        E = e["t"] - e["start"]
        cand = [s for s in steps[e["video_id"]] if s[0] != e["step"] and s[2] - s[1] > E + 20]
        if cand:
            si, st, en, ins, act = rng.choice(cand)
            tt = st + (urng.uniform(5.0, en - st - 20.0) if unmatched else E)
            add.append(dict(e, kind="notdone", step=si, start=st, end=en, instruction=ins, t=tt, target=SILENCE, cat=None))
    ex += add
    if comist and split == "train":
        from huggingface_hub import hf_hub_download
        rows = {r["video_id"]: r for r in load_dataset(HF, plan_set, split=split).to_list()}
        cm = json.load(open(hf_hub_download(COMIST, "annotations.json", repo_type="dataset")))
        allfb = collections.defaultdict(list)
        for act, v in bank.items():
            for x, c in v: allfb[act].append((x, c))
        for x in cm:
            v = x["video_id"]
            if x["dataset"] != "CaptainCook4D" or v not in rows or not frame_dir(frames_root, v) or (ids and v not in ids): continue
            act = rows[v]["activity_name"]; st = x["start_time"]; f = x["text_timestamps"][1]; fb = x["texts"][1]
            c = x["mistake_type"].replace("_error", "")
            base = {"video_id": v, "activity": act, "step": -1, "instruction": x["texts"][0], "start": st, "end": x["end_time"], "src": "comist"}
            hard = [h for h in rng.sample(allfb[act], min(12, len(allfb[act])))]
            for dt in (0, 5, 10):
                ex.append(dict(base, kind="mistake", t=f + dt, target=fb, cat=c, hard=[h for h in hard if h[0] != fb]))
            t = f - 20 - rng.uniform(0, 10)
            if t >= st + 5:
                ex.append(dict(base, kind="shifted", t=t, target=SILENCE, cat=None, hard=[(fb, c)] + hard[:11]))
            bank.setdefault(act, []); bank[act] = sorted(set(bank[act]) | {(fb, c)})
    # (c) elapsed-matched 'clean' negatives for the mistake prompt: for every mistake situation at elapsed E, one
    #     situation at elapsed E in a mistake-free GT step of the same video with >= 15 s left (target silence)
    clean = collections.defaultdict(list)
    for e in ex:
        if e["kind"] in ("silence", "success") and e["step"] >= 0: clean[e["video_id"]].append((e["step"], e["start"], e["end"], e["instruction"]))
    mfree = {v: sorted(set(x for x in L)) for v, L in clean.items()}
    gt_mis_steps = {(e["video_id"], e["step"]) for e in ex if e["kind"] == "mistake" and e.get("src") != "comist"}
    add = []
    for e in ex:
        if e["kind"] != "mistake": continue
        E = e["t"] - e["start"]
        cand = [s for s in mfree.get(e["video_id"], []) if (e["video_id"], s[0]) not in gt_mis_steps and s[2] - s[1] > E + 15]
        if cand:
            si, st, en, ins = rng.choice(cand)
            tt = st + (urng.uniform(5.0, en - st - 15.0) if unmatched else E)
            add.append(dict(e, kind="clean", step=si, start=st, end=en, instruction=ins, t=tt, target=SILENCE, cat=None, src="unmatched" if unmatched else "matched"))
    ex += add
    return ex, bank


def qicd_style(s):
    """CoMist instruction -> QICD surface form: 'Now ' + lower-cased first letter, trailing period."""
    s = s.strip()
    s = "Now " + s[:1].lower() + s[1:]
    return s if s.endswith(".") else s + "."


def build_v4(frames_root, split="train", plan_set="advanced_planning", stride=20.0, ids=None, seed=0):
    """Round-4 situations = build_v2 (QICD) + elapsed-matched 'notdone' and 'clean' negatives + Ego-CoMist pairs with
    (i) the counterfactual instruction rewritten to QICD form (qicd_style) and (ii) a PAIRED negative for every
    CoMist positive: same video, same window start, same times, but the ORIGINAL QICD instruction of the step the
    clip lies in (kind 'paired', target silence / "NO." under the mistake prompt). Only the instruction differs.
    CoMist pairs whose clip does not overlap a mistake-free QICD step by >= 50% are dropped."""
    ex, bank = build_v3(frames_root, split, plan_set, stride, ids, seed, comist=False)
    if split != "train":
        return ex, bank
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed + 11)
    rows = {r["video_id"]: r for r in load_dataset(HF, plan_set, split=split).to_list()}
    segs = {v: segments(r) for v, r in rows.items()}
    cm = json.load(open(hf_hub_download(COMIST, "annotations.json", repo_type="dataset")))
    allfb = collections.defaultdict(list)
    for act, v in bank.items():
        for x, c in v: allfb[act].append((x, c))
    kept = dropped = 0
    for x in cm:
        v = x["video_id"]
        if x["dataset"] != "CaptainCook4D" or v not in rows or not frame_dir(frames_root, v) or (ids and v not in ids): continue
        st, en = x["start_time"], x["end_time"]; best, ov = None, 0.0
        for s in segs[v]:
            a0, a1 = s["ts"][0], max(s["ts"]); o = max(0.0, min(en, a1) - max(st, a0)) / max(en - st, 1e-6)
            if o > ov: best, ov = s, o
        if best is None or ov < 0.5 or best["has_mistake"]:
            dropped += 1; continue
        kept += 1
        act = rows[v]["activity_name"]; f = x["text_timestamps"][1]; fb = x["texts"][1]; c = x["mistake_type"].replace("_error", "")
        cf, orig = qicd_style(x["texts"][0]), best["texts"][0]
        base = {"video_id": v, "activity": act, "step": -1, "start": st, "end": en, "src": "comist"}
        hard = rng.sample(allfb[act], min(12, len(allfb[act])))
        for dt in (0, 5, 10):
            ex.append(dict(base, kind="mistake", instruction=cf, t=f + dt, target=fb, cat=c, hard=[h for h in hard if h[0] != fb]))
            ex.append(dict(base, kind="paired", instruction=orig, t=f + dt, target=SILENCE, cat=None, hard=[(fb, c)] + hard[:11]))
        t = f - 20 - rng.uniform(0, 10)
        if t >= st + 5:
            ex.append(dict(base, kind="shifted", instruction=cf, t=t, target=SILENCE, cat=None, hard=[(fb, c)] + hard[:11]))
        bank.setdefault(act, []); bank[act] = sorted(set(bank[act]) | {(fb, c)})
    # elapsed-matched clean negatives for the QICD mistakes only (CoMist positives already have their paired negative)
    print(f"build_v4: CoMist pairs kept {kept}, dropped {dropped} (no >=50% overlap with a mistake-free QICD step)", flush=True)
    return ex, bank


def shortcut_audit(ex):
    """Instruction-text-only classifier for the mistake-prompt target (mistake vs every other situation answered with
    the mistake prompt). Hashed unigrams+bigrams of the instruction + length + 'starts with Now'; recipe-grouped 5-fold.
    AUROC near 0.5 = the label cannot be read from the instruction alone."""
    import numpy as np, torch
    E = [e for e in ex if e["kind"] in ("mistake", "clean", "paired", "silence", "shifted")]
    y = np.array([e["kind"] == "mistake" for e in E], dtype=np.float32)
    D = 4096; X = np.zeros((len(E), D + 2), np.float32)
    for i, e in enumerate(E):
        w = e["instruction"].lower().replace(".", " .").split()
        for t in w + [a + "_" + b for a, b in zip(w, w[1:])]: X[i, hash(t) % D] += 1
        X[i, D] = len(w) / 20; X[i, D + 1] = float(e["instruction"].startswith("Now"))
    g = np.array([e["video_id"].split("_")[0] for e in E]); rec = sorted(set(g)); fold = {r: i % 5 for i, r in enumerate(rec)}
    f = np.array([fold[r] for r in g]); sc = np.zeros(len(E))
    for k in range(5):
        tr, te = f != k, f == k; A = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
        w_ = torch.zeros(D + 2, requires_grad=True); b = torch.zeros(1, requires_grad=True); opt = torch.optim.Adam([w_, b], 0.05)
        for _ in range(200):
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(A @ w_ + b, yt) + 1e-3 * (w_ ** 2).sum(); l.backward(); opt.step()
        sc[te] = (torch.tensor(X[te]) @ w_ + b).detach().numpy()
    p, n = sc[y == 1], sc[y == 0]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def norm_instr(s):
    return " ".join(s.strip().lower().rstrip(".").split())


def pair_partners(ex, tol_elapsed=10.0, tol_remaining=2.5):
    """Round-5 cross-participant pairs. For every QICD mistake situation A (video v, step instruction I, elapsed
    E = t - start, remaining R = end - t; R <= 0 for 90% of them because a mistake step's GT end is its last
    feedback) the partners are the same-instruction steps of OTHER videos of the same recipe whose GT step has no
    mistake (identified by having a 'success' situation). Partner time t' = end' - R (same task phase), clipped to
    [start' + 5, end' + 5]; kept only if |elapsed difference| <= tol_elapsed and |remaining difference| <= tol_remaining,
    so neither observable elapsed time nor phase-of-step separates the pair (audit: both ~0.5). The partner uses A's
    exact instruction string, so the pair differs only in the video. All situations come from ex's split (train)."""
    clean = collections.defaultdict(dict)       # (activity, norm instr) -> {(video, step): (start, end)}
    for e in ex:
        if e["kind"] == "success" and e["step"] >= 0 and e.get("src") is None:
            clean[(e["activity"], norm_instr(e["instruction"]))][(e["video_id"], e["step"])] = (e["start"], e["end"])
    out = {}
    for i, e in enumerate(ex):
        if e["kind"] != "mistake" or e.get("src") == "comist": continue
        E, R = e["t"] - e["start"], e["end"] - e["t"]; P = []
        for (v, si), (st, en) in sorted(clean.get((e["activity"], norm_instr(e["instruction"])), {}).items()):
            if v == e["video_id"]: continue
            t = min(max(en - R, st + 5), en + 5)
            if abs(E - (t - st)) > tol_elapsed or abs(R - (en - t)) > tol_remaining: continue
            P.append(dict(video_id=v, activity=e["activity"], step=si, instruction=e["instruction"], start=st, end=en, t=t,
                          kind="pair_end" if t >= en else "pair_mid", target=SILENCE, cat=None, hard=[]))
        if P: out[i] = P
    return out


def pair_audit(ex, partners, other_split_ids=()):
    """Shortcut / leakage audit for the round-5 pairs (CPU)."""
    import statistics as stt
    A = [(ex[i], p) for i, P in partners.items() for p in P]
    vids_used = {e["video_id"] for e, _ in A} | {p["video_id"] for _, p in A}
    leak = vids_used & set(other_split_ids)
    dE = [abs((e["t"] - e["start"]) - (p["t"] - p["start"])) for e, p in A]
    sE = [(e["t"] - e["start"]) - (p["t"] - p["start"]) for e, p in A]; sR = [(p["end"] - p["t"]) - (e["end"] - e["t"]) for e, p in A]
    rk = lambda L: sum((x > 1e-6) + 0.5 * (abs(x) <= 1e-6) for x in L) / len(L)
    win = lambda x: min(x["t"] - x["start"], 120.0)
    dW = [abs(win(e) - win(p)) for e, p in A]
    asA = collections.Counter(e["video_id"] for e, _ in A); asB = collections.Counter(p["video_id"] for _, p in A)
    both = sum(1 for v in vids_used if asA[v] and asB[v])
    n_mis = sum(1 for e in ex if e["kind"] == "mistake" and e.get("src") is None)
    return {"mistake_situations": n_mis, "with_partner": len(partners), "pairs": len(A),
            "partner_kind": dict(collections.Counter(p["kind"] for _, p in A)),
            "same_instruction_string": all(e["instruction"] == p["instruction"] for e, p in A),
            "elapsed_absdiff_median": stt.median(dE), "elapsed_only_pair_acc": rk(sE), "phase_only_pair_acc": rk(sR), "elapsed_matched_exact_frac": sum(d < 1e-6 for d in dE) / len(dE),
            "window_absdiff_median": stt.median(dW), "videos": len(vids_used), "videos_as_both_A_and_B": both,
            "leak_videos_from_other_splits": sorted(leak)}
