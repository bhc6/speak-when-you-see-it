"""SFT (+ Situation-Intervention Alignment auxiliary loss) LoRA training of Qwen3-VL-2B on QICD situations.

Generation target uses the official baseline prompts unchanged (ai_coach_cooking_2026/qwen3_vl_baseline.py), so a
trained adapter is evaluated with the official turn-based protocol by qic_lora.py:
  completion prompt  ->  "YES." for success situations, "NO." otherwise
  mistake prompt     ->  "YES. <GT feedback>" for mistake situations, "NO." otherwise (silence / success / shifted)
Each step draws one of the two prompts per situation (success/mistake situations prefer the informative prompt).

SIA (when --lambda_sia > 0): situation vector = hidden state of the last prompt token at --layer, through a small
MLP head; candidate vectors = the same backbone run text-only on 'The cooking assistant says to the person: "<c>"'
(no grad), last token at --layer, through a second MLP head. Candidates per situation: the target, success, silence,
the situation's hard negatives, random bank feedbacks, and a FIFO queue of earlier candidate states (re-projected
with the current head). Loss: SigLIP pairwise sigmoid with learnable temperature/bias; identical texts to the
target are masked out as false negatives.

  python sia_train.py --frames_root R --save_dir D [--lambda_sia 0.5] [--overfit 32] [--steps N] [--mem_test]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("QIC_BASELINE_DIR", os.path.join(HERE, "..", "ref_sys", "ai_coach_cooking_2026"))
sys.path.insert(0, os.path.abspath(BASE))

import qwen3_vl_baseline as baseline  # noqa: E402
import sia_data  # noqa: E402
import vlm_common as V  # noqa: E402
from utils import load_frames_into_array  # noqa: E402  (quick-start repo)

CAND_TMPL = 'The cooking assistant says to the person: "{}"'


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--ids", default=None, help="comma-separated video ids (default: all train videos with frames)")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--lambda_sia", type=float, default=0.0)
    p.add_argument("--lambda_sft", type=float, default=1.0, help="0 = SIA-only ablation")
    p.add_argument("--v2", action="store_true", help="round-2 recipe: balanced situations/prompts, two-level SIA")
    p.add_argument("--v3", action="store_true", help="round-3 data: v2 + elapsed-matched not-done negatives + Ego-CoMist (implies --v2)")
    p.add_argument("--no_comist", action="store_true", help="v3 without Ego-CoMist (round 4: text-shortcut free)")
    p.add_argument("--head_type", default="mlp", choices=["mlp", "linear"])
    p.add_argument("--type_balance", action="store_true", help="type-level CE weighted to equalise silence/success/mistake")
    p.add_argument("--snapshot_steps", default="", help="comma list: also keep adapter_s<N> at these steps")
    p.add_argument("--lambda_speak", type=float, default=0.0, help=">0: T3 speak/silent head (BCE) on the same hidden state")
    p.add_argument("--lambda_dpo", type=float, default=0.0, help=">0: T4 DPO on the generated answer, reference = adapter off")
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--drop_kinds", default="", help="comma list of situation kinds removed from the training data "
                   "(ablation of the evidence-contrastive negatives: notdone = time-matched not-yet-done for completion, "
                   "clean = time-matched mistake-free for mistakes). Sampling weights of the remaining kinds are unchanged.")
    p.add_argument("--kind_weight", default="", help="override sampling weights, e.g. clean=0.075 (dose-response of a kind; other weights unchanged, the sampler normalises)")
    p.add_argument("--frame_cap", type=int, default=32, help="non-Qwen3-VL backbones only: uniform temporal subsampling of the "
                   "window to at most N frames (same as qic_vlm.py at evaluation)")
    p.add_argument("--black_frames", action="store_true", help="control (blind ECT): every input frame is replaced by an all-black "
                   "frame of the same size and count; data, negatives, prompts, steps unchanged -> the model sees only text, elapsed time "
                   "(via window length) and nothing of the video")
    p.add_argument("--fixed_window", type=int, default=0, help="appendix control: train on the last N frames ending at t "
                   "(crossing step boundaries) instead of [step start, t]; 0 = official growing window")
    p.add_argument("--unmatched_negatives", action="store_true", help="control: notdone/clean negatives at a uniformly "
                   "drawn time inside the same partner steps instead of the positive's elapsed time (counts unchanged)")
    p.add_argument("--pair", action="store_true", help="round 5: every QICD mistake situation is trained together with a "
                   "cross-participant partner (same recipe + instruction, other video, mistake-free step, elapsed-matched)")
    p.add_argument("--lambda_pair", type=float, default=0.0, help=">0: pairwise ranking on the mistake-prompt yes/no "
                   "log-odds, z(mistake video) > z(partner); 0 = data/compute-matched SFT control")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=0, help="optimizer steps (0 = 2 epochs)")
    p.add_argument("--max_buffer", type=int, default=240)
    p.add_argument("--n_cand", type=int, default=32)
    p.add_argument("--sia_loss", default="sigmoid", choices=["sigmoid", "ce"])
    p.add_argument("--head_lr_mult", type=float, default=5.0)
    p.add_argument("--head_lr", type=float, default=0.0, help="absolute head lr (overrides mult)")
    p.add_argument("--debug_eval_every", type=int, default=0, help="eval SIA acc on all situations in eval mode")
    p.add_argument("--queue", type=int, default=2048)
    p.add_argument("--overfit", type=int, default=0, help="train on the first N situations only")
    p.add_argument("--mem_test", action="store_true", help="one fwd/bwd on the longest window, report memory, exit")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="flash_attention_2")
    return p.parse_args()


class Head(nn.Module):
    """(Debug note 2026-11-03: raw last-token states of different situations have mean pairwise cosine 0.997; after
    per-dimension standardisation 0.002 and a linear head separates 32 situations perfectly. LoRA dropout noise is
    larger than this signal, so LoRA dropout is 0.)
    Per-dimension standardisation with statistics estimated once before training (calibrate), then an MLP.
    Situation states at the last prompt token differ across inputs only by a small fraction of their norm (shared
    token/position component + outlier dimensions); without per-dimension centring every situation projects to
    nearly the same point (observed: the head predicts the majority candidate for all inputs)."""
    def __init__(self, d=2048, k=512, linear=False):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))
        self.net = nn.Linear(d, k) if linear else nn.Sequential(nn.Linear(d, k), nn.GELU(), nn.Linear(k, k))

    def calibrate(self, X):
        self.mu.copy_(X.float().mean(0)); self.sd.copy_(X.float().std(0) + 1e-3)

    def forward(self, x):
        x = x.float()
        if self.training:  # track drift of the backbone states under LoRA (EMA of per-dimension mean/std)
            with torch.no_grad():
                xs = x.reshape(-1, x.shape[-1])
                self.mu.mul_(0.99).add_(0.01 * xs.mean(0))
                if xs.shape[0] > 1: self.sd.mul_(0.99).add_(0.01 * (xs.std(0) + 1e-3))
        return F.normalize(self.net((x - self.mu) / self.sd), dim=-1)


def lm_layers(model):
    m = model.base_model.model if hasattr(model, "base_model") else model
    return m.model.language_model


def main():
    a = args_()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    mode_rng = random.Random(a.seed + 2)
    os.makedirs(a.save_dir, exist_ok=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ids = set(a.ids.split(",")) if a.ids else None
    if a.v3: a.v2 = True
    if a.v2:
        ex, bank = (sia_data.build_v3(a.frames_root, "train", ids=ids, seed=a.seed, comist=not a.no_comist, unmatched=a.unmatched_negatives) if a.v3
                    else sia_data.build_v2(a.frames_root, "train", ids=ids, seed=a.seed))
        cat_of = {x: c for v in bank.values() for x, c in v}
        bank = {k: [x for x, _ in v] for k, v in bank.items()}
        for e in ex: e["hard"] = [h[0] for h in e["hard"]]
        by_kind = {k: [i for i, e in enumerate(ex) if e["kind"] == k] for k in ("silence", "success", "mistake", "shifted", "notdone", "clean")}
        KW = ({"silence": 0.15, "notdone": 0.15, "clean": 0.15, "success": 0.20, "mistake": 0.25, "shifted": 0.10} if a.v3
              else {"silence": 0.40, "success": 0.25, "mistake": 0.25, "shifted": 0.10})
        for kv in filter(None, a.kind_weight.split(",")):
            k_, v_ = kv.split("="); assert k_ in KW, k_; KW[k_] = float(v_)
        if a.kind_weight: print(f"sia: kind weights {KW}", flush=True)
        _share = [sum(v for k, v in KW.items() if k not in ("success", "mistake")), KW["success"], KW["mistake"]]
        type_w = (torch.tensor([1 / x for x in _share], device="cuda") / sum(1 / x for x in _share) * 3) if a.type_balance else None
    else:
        ex, bank = sia_data.build(a.frames_root, "train", ids=ids, seed=a.seed)
    if a.drop_kinds:
        _drop = set(a.drop_kinds.split(",")); _n = len(ex)
        ex = [e for e in ex if e["kind"] not in _drop]
        if a.v2:
            by_kind = {k: [i for i, e in enumerate(ex) if e["kind"] == k] for k in by_kind}
        print(f"sia: drop_kinds {sorted(_drop)}: {_n} -> {len(ex)} situations", flush=True)
    if a.overfit: ex = ex[:a.overfit]
    partners = {}
    if a.pair:
        partners = sia_data.pair_partners(ex)
        from datasets import load_dataset as _ld
        other = {r["video_id"] for sp in ("validation", "test") for r in _ld(sia_data.HF, "advanced_planning", split=sp).to_list()}
        aud = sia_data.pair_audit(ex, partners, other)
        print("sia: pair audit", json.dumps(aud), flush=True)
        assert not aud["leak_videos_from_other_splits"] and aud["same_instruction_string"]
    pair_rng = random.Random(a.seed + 3)
    all_fb = sorted({x for v in bank.values() for x in v})
    print(f"sia: situations {len(ex)} kinds {dict(__import__('collections').Counter(e['kind'] for e in ex))} "
          f"videos {len({e['video_id'] for e in ex})} bank {len(all_fb)}", flush=True)

    FAM = V.family(a.model_id)
    model = V.model_class(FAM).from_pretrained(a.model_id, dtype=torch.bfloat16, attn_implementation=a.attn if FAM == "qwen3vl" else "sdpa",
                                               device_map="cuda:0")
    proc = V.fix_processor(AutoProcessor.from_pretrained(a.model_id), a.model_id, FAM)
    tok = proc.tokenizer
    names = sorted({n for n, _ in model.named_modules() if ".language_model.layers." in n and
                    n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")})
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules=names))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads(); model.config.use_cache = False
    model.print_trainable_parameters()
    vis = V.vision_module(model)
    _vis_fwd = vis.forward
    def _vis_nograd(*x, **k):
        with torch.no_grad():
            return _vis_fwd(*x, **k)
    vis.forward = _vis_nograd
    hs, hc = Head(linear=a.head_type == "linear").cuda(), Head(linear=a.head_type == "linear").cuda()
    protos = nn.Parameter(F.normalize(torch.randn(3, 512, device="cuda"), dim=-1))  # v2 type level: silence/success/mistake
    speak = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 1)).cuda()
    _sp = speak.forward; speak.forward = lambda x: _sp(x)[..., 0]
    logit_t = nn.Parameter(torch.tensor(np.log(10.0), device="cuda")); logit_b = nn.Parameter(torch.tensor(-10.0, device="cuda"))
    params = [p for p in model.parameters() if p.requires_grad]
    extra = list(hs.parameters()) + list(hc.parameters()) + [logit_t, logit_b] + list(speak.parameters()) + [protos]
    opt = torch.optim.AdamW([{"params": params}, {"params": extra, "lr": a.head_lr or a.lr * a.head_lr_mult}], lr=a.lr, weight_decay=0.0)
    steps = a.steps or max(1, 2 * len(ex) // a.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    YES_ID, NO_ID = tok.encode("YES", add_special_tokens=False)[0], tok.encode("NO", add_special_tokens=False)[0]
    if FAM == "qwen3vl" or a.pair:
        assert tok.encode("YES.", add_special_tokens=False)[0] == YES_ID and tok.encode("NO.", add_special_tokens=False)[0] == NO_ID
    frames_cache = {}
    queue_h, queue_txt = [], []
    start_step, st = 0, {}
    ck = os.path.join(a.save_dir, "last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(st["lora"], strict=False); hs.load_state_dict(st["hs"]); hc.load_state_dict(st["hc"])
        if "speak" in st: speak.load_state_dict(st["speak"])
        if "protos" in st: protos.data.copy_(st["protos"])
        logit_t.data, logit_b.data = st["t"], st["b"]; opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_step = st["step"]; print(f"sia: resumed at step {start_step}", flush=True)

    def video(vid):
        if vid not in frames_cache:
            d = sia_data.frame_dir(a.frames_root, vid)
            fs = sorted((f for f in os.listdir(d) if f.endswith(".jpg")), key=lambda x: int(x.split("_")[1].split(".")[0]))
            frames_cache.clear()
            frames_cache[vid] = [os.path.join(d, f) for f in fs]
        return frames_cache[vid]

    def situation_inputs(e, mode, target=None):
        paths = video(e["video_id"])
        i0 = int(e["start"] * 2); seek = min(max(int(e["t"] * 2), i0 + 10), len(paths)); b0 = max(seek - a.max_buffer, i0)
        if a.fixed_window: b0 = max(seek - a.fixed_window, 0)
        clip = load_frames_into_array(paths[b0:seek], video_input_resolution=(640, 360))
        if a.black_frames: clip = np.zeros_like(clip)
        if target is None:
            target = gold(e, mode)
        msgs = V.messages(baseline.build_messages(mode, instruction=e["instruction"]), FAM)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + target + "<|im_end|>\n"
        inp = V.encode(proc, FAM, text, clip, a.frame_cap)
        if FAM != "qwen3vl":
            for k_, v_ in inp.items():
                if torch.is_floating_point(v_): inp[k_] = v_.to(torch.bfloat16)
        ids_ = inp["input_ids"][0]
        a_start = int((ids_ == im_start).nonzero()[-1]) + 3          # <|im_start|> assistant \n
        if not getattr(situation_inputs, "_checked", False):   # label mask must start exactly at the answer
            assert tok.decode(ids_[a_start:]).startswith(target.split()[0]), (FAM, tok.decode(ids_[a_start - 4:a_start + 4]))
            situation_inputs._checked = True
        labels = ids_.clone(); labels[:a_start] = -100
        inp["labels"] = labels[None]
        return inp.to("cuda"), a_start - 1, seek - b0, len(ids_) - a_start

    def gold(e, mode):
        if mode == "instruction_end":
            return "YES." if e["kind"] == "success" else "NO."
        return f"YES. {e['target']}" if e["kind"] == "mistake" else "NO."

    def rejected(e, mode):
        if mode == "instruction_end":
            return "NO." if e["kind"] == "success" else "YES."
        wrong = [h for h in e["hard"] if h != e["target"]]
        if e["kind"] == "mistake" and (not wrong or random.random() < 0.5):
            return "NO."
        return f"YES. {random.choice(wrong) if wrong else 'You made a mistake.'}"

    def answer_logp(e, mode, target, with_grad=True, ref=False):
        inp, pos, _, n_ans = situation_inputs(e, mode, target); labels = inp.pop("labels")[0]
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            if ref:
                with model.disable_adapter():
                    out = model(**inp, use_cache=False, logits_to_keep=n_ans + 1)
            else:
                out = model(**inp, use_cache=False, logits_to_keep=n_ans + 1)
        lp = torch.log_softmax(out.logits[0, :-1].float(), -1).gather(1, labels[-n_ans:, None])[:, 0]
        return lp.sum(), lp.mean()

    @torch.no_grad()
    def encode_reply(texts):
        """v2 candidates: each text as the assistant's own reply in the official output format ("YES. <feedback>")
        under the same chat template; mean of the reply tokens' states at --layer (no grad). Same 'answer space' the
        situation's last prompt token predicts."""
        convs = [proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "."}]}], tokenize=False,
                                          add_generation_prompt=True) for _ in texts]
        full = [c + f"YES. {t}" for c, t in zip(convs, texts)]
        enc = tok(full, return_tensors="pt", padding=True, padding_side="right").to("cuda")
        plen = len(tok(convs[0])["input_ids"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lm_layers(model)(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[a.layer].float(); m = enc.attention_mask.clone(); m[:, :plen] = 0
        return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)

    @torch.no_grad()
    def encode_text(texts):
        enc = tok([proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": CAND_TMPL.format(t)}]}],
                                            tokenize=False, add_generation_prompt=True) for t in texts],
                  return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lm_layers(model)(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
        last = enc.attention_mask.sum(1) - 1
        return out.hidden_states[a.layer][torch.arange(len(texts)), last].float()

    def step_one(e):
        # prompt choice from its own RNG, consumed identically in every run -> T1/S1/T3/... see the same (situation,
        # prompt) sequence; method-specific sampling (candidates, rejected answers) uses the global RNG
        r1, r2 = mode_rng.random(), mode_rng.random()
        mode = ("mistake_inference" if e["kind"] in ("mistake", "clean") else "instruction_end" if e["kind"] in ("success", "notdone")
                else ["instruction_end", "mistake_inference"][int(r1 < 0.5)])
        if e["kind"] in ("mistake", "success", "notdone", "clean") and r2 < (0.2 if a.v2 else 0.3):
            mode = "instruction_end" if mode == "mistake_inference" else "mistake_inference"
        if os.environ.get("SIA_TRACE"): print(f"trace {e['video_id']} {e['t']:.1f} {e['kind']} {mode}", flush=True)
        if a.lambda_dpo > 0:  # T4: SFT on chosen + DPO(chosen vs rejected), reference = base model (adapter off)
            ch, rj = gold(e, mode), rejected(e, mode)
            pc, pc_mean = answer_logp(e, mode, ch); pr, _ = answer_logp(e, mode, rj)
            rc, _ = answer_logp(e, mode, ch, False, True); rr, _ = answer_logp(e, mode, rj, False, True)
            loss_dpo = -F.logsigmoid(a.dpo_beta * ((pc - pr) - (rc - rr)))
            loss = a.lambda_sft * (-pc_mean) + a.lambda_dpo * loss_dpo
            (loss / a.grad_accum).backward()
            return float(-pc_mean), float(loss_dpo), float(pc - pr > rc - rr), 0, 0, mode, e["kind"]
        inp, pos, nframes, n_ans = situation_inputs(e, mode)
        labels = inp.pop("labels")[0]
        need_h = a.lambda_sia > 0 or a.lambda_speak > 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inp, output_hidden_states=need_h, use_cache=False, logits_to_keep=n_ans + 1)
        lg = out.logits[0, :-1].float(); tgt = labels[-n_ans:]
        loss_sft = F.cross_entropy(lg, tgt)
        loss_speak = torch.zeros((), device="cuda")
        if a.lambda_speak > 0:  # T3: binary speak / stay-silent head (Streaming Interventions-style action head)
            yk = torch.tensor(float(e["kind"] in ("mistake", "success")), device="cuda")
            loss_speak = F.binary_cross_entropy_with_logits(speak(out.hidden_states[a.layer][0, pos].float()), yk)
        loss_sia = torch.zeros((), device="cuda"); acc = float("nan")
        if a.lambda_sia > 0 and a.v2:
            g_s = hs(out.hidden_states[a.layer][0, pos])
            ty = {"silence": 0, "shifted": 0, "notdone": 0, "clean": 0, "success": 1, "mistake": 2}[e["kind"]]
            zt = (F.normalize(protos, dim=-1) @ g_s) * logit_t.exp()
            loss_sia = F.cross_entropy(zt[None], torch.tensor([ty], device="cuda"), weight=type_w)   # type level
            acc = float(zt.argmax() == ty)
            if e["kind"] == "mistake":  # content level: SupCon, positives = same error category (other videos/steps too)
                cands = list(dict.fromkeys([e["target"]] + e["hard"] + random.sample(all_fb, min(len(all_fb), a.n_cand))))[:a.n_cand]
                pos_m = torch.tensor([c == e["target"] or cat_of.get(c) == e["cat"] for c in cands], device="cuda")
                zc = (hc(encode_reply(cands)) @ g_s) * logit_t.exp()
                loss_c = torch.logsumexp(zc, 0) - torch.logsumexp(zc[pos_m], 0)
                loss_sia = loss_sia + loss_c
        elif a.lambda_sia > 0:
            cands = [e["target"], sia_data.SUCCESS, sia_data.SILENCE] + [h for h in e["hard"] if h != e["target"]]
            pool = [x for x in all_fb if x != e["target"] and x not in cands]
            cands += random.sample(pool, max(0, min(len(pool), a.n_cand - len(cands))))
            cands = list(dict.fromkeys(cands))
            h_c = encode_text(cands)
            if queue_h:
                qh = torch.stack(queue_h); keep = [i for i, t in enumerate(queue_txt) if t != e["target"]]
                h_all = torch.cat([h_c, qh[keep]]); n_q = len(keep)
            else:
                h_all, n_q = h_c, 0
            g_s = hs(out.hidden_states[a.layer][0, pos]); g_c = hc(h_all)
            z = (g_c @ g_s) * logit_t.exp() + logit_b
            y = -torch.ones_like(z); y[0] = 1.0
            if a.sia_loss == "ce":
                loss_sia = F.cross_entropy((g_c @ g_s * logit_t.exp())[None], torch.zeros(1, dtype=torch.long, device="cuda"))
            else:
                loss_sia = -F.logsigmoid(y * z).sum() / 1.0
            acc = float(z[:len(cands)].argmax() == 0)
            for i in range(len(cands)):
                queue_h.append(h_c[i].detach()); queue_txt.append(cands[i])
            del queue_h[:-a.queue]; del queue_txt[:-a.queue]
        loss = a.lambda_sft * loss_sft + a.lambda_sia * loss_sia + a.lambda_speak * loss_speak
        (loss / a.grad_accum).backward()
        return float(loss_sft), float(loss_sia), acc, nframes, inp["input_ids"].shape[1], mode, e["kind"]

    def sft_z(e, mode, grad=True):
        """SFT loss and first-answer-token log-odds z = logit(YES) - logit(NO) for the official prompt."""
        inp, pos, nframes, n_ans = situation_inputs(e, mode); labels = inp.pop("labels")[0]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inp, use_cache=False, logits_to_keep=n_ans + 1)
            lg = out.logits[0, :-1].float()
            loss_sft = F.cross_entropy(lg, labels[-n_ans:])
            z = lg[0, YES_ID] - lg[0, NO_ID]
        return loss_sft, z, nframes, inp["input_ids"].shape[1]

    def pair_step(e, p):
        """Mistake situation e and its cross-participant partner p, both under the official mistake prompt (targets
        "YES. <feedback>" / "NO."). Loss = SFT(e) + SFT(p) + lambda_pair * -log sigmoid(z_e - z_p). The two graphs are
        never held together: z_p is first computed without grad, e is back-propagated against the constant z_p, then p
        against the constant z_e -> exact gradient of the summed loss at single-situation memory."""
        mode_rng.random(); mode_rng.random()          # keep the prompt RNG stream aligned with step_one
        mode = "mistake_inference"
        zp0 = sft_z(p, mode, grad=False)[1].detach() if a.lambda_pair > 0 else None
        le, ze, nf, nt = sft_z(e, mode)
        lp_e = -F.logsigmoid(ze - zp0) if a.lambda_pair > 0 else torch.zeros((), device="cuda")
        ((a.lambda_sft * le + a.lambda_pair * lp_e) / a.grad_accum).backward()
        lpp, zp, nf2, nt2 = sft_z(p, mode)
        lp_p = -F.logsigmoid(ze.detach() - zp) if a.lambda_pair > 0 else torch.zeros((), device="cuda")
        ((a.lambda_sft * lpp + a.lambda_pair * lp_p) / a.grad_accum).backward()
        acc = float(ze.detach() > zp.detach()); lpair = float(-F.logsigmoid(ze.detach() - zp.detach()))
        return [(float(le), lpair, acc, nf, nt, mode, "mistake"), (float(lpp), lpair, acc, nf2, nt2, mode, p["kind"])]

    if a.lambda_sia > 0 and not os.path.exists(ck):
        model.eval(); H = []
        with torch.no_grad():
            for e in random.sample(ex, min(64, len(ex))):
                inp, pos, _, n_ans = situation_inputs(e, random.choice(["instruction_end", "mistake_inference"])); inp.pop("labels")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(**inp, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                H.append(o.hidden_states[a.layer][0, pos].float())
            hs.calibrate(torch.stack(H))
            C = random.sample(all_fb, min(256, len(all_fb))) if a.v2 else [sia_data.SUCCESS, sia_data.SILENCE] + random.sample(all_fb, min(254, len(all_fb)))
            enc_fn = encode_reply if a.v2 else encode_text
            hc.calibrate(torch.cat([enc_fn(C[i:i + 32]) for i in range(0, len(C), 32)]))
        print(f"sia: calibrated heads on {len(H)} situations / {len(C)} candidates", flush=True)
    model.train()  # HF activates gradient checkpointing only in training mode
    if a.mem_test:
        e = max(ex, key=lambda e: min(e["t"] - e["start"], a.max_buffer / 2))
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        r = step_one(dict(e, kind="mistake" if e["kind"] == "mistake" else e["kind"]))
        torch.cuda.synchronize()
        print(f"sia mem_test: frames {r[3]} tokens {r[4]} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"time {time.time() - t0:.1f}s loss_sft {r[0]:.3f} loss_sia {r[1]:.3f}", flush=True)
        return

    order, order_rng = [], random.Random(a.seed + 1)
    if start_step and "rng" in st:
        order_rng.setstate(st["rng"]["order"]); order = st["rng"]["order_list"]; mode_rng.setstate(st["rng"]["mode"]); random.setstate(st["rng"]["py"])
        if "pair" in st["rng"]: pair_rng.setstate(st["rng"]["pair"])  # own RNG: SFT and SFT+SIA see the same situation order
    log = open(os.path.join(a.save_dir, "train_log.jsonl"), "a")
    model.train(); t0 = time.time(); stats = []
    for s in range(start_step, steps):
        for _ in range(a.grad_accum):
            if not order and os.environ.get("SIA_PAIR_SMOKE"):
                order = sorted(partners)[:64]
            if not order:
                if a.v2:
                    ks = list(KW); order = [order_rng.choice(by_kind[k]) for k in order_rng.choices(ks, [KW[k] for k in ks], k=4096) if by_kind[k]]
                else:
                    order = order_rng.sample(range(len(ex)), len(ex))
            i = order.pop()
            if a.pair and i in partners:
                stats.extend(pair_step(ex[i], pair_rng.choice(partners[i])))
            else:
                stats.append(step_one(ex[i]))
        torch.nn.utils.clip_grad_norm_(params + extra, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % 5 == 0 or s == steps - 1:
            S = np.array([x[:4] for x in stats], dtype=float)
            rec = {"step": s + 1, "loss_sft": S[:, 0].mean(), "loss_sia": S[:, 1].mean(), "acc_sia": np.nanmean(S[:, 2]) if (a.lambda_sia or a.lambda_pair) else None,
                   "sec_per_situation": (time.time() - t0) / len(stats),
                   "acc_by_kind": {k: round(float(np.nanmean([x[2] for x in stats if x[6] == k])), 2) for k in sorted({x[6] for x in stats})} if (a.lambda_sia or a.lambda_pair) else None, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "lr": sched.get_last_lr()[0]}
            print("sia:", json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            stats, t0 = [], time.time()
        if a.debug_eval_every and (s + 1) % a.debug_eval_every == 0:
            model.eval(); hs.eval(); hc.eval(); ok = []
            with torch.no_grad():
                for e in ex:
                    inp, pos, _, n = situation_inputs(e, "mistake_inference"); inp.pop("labels")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model(**inp, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                    cands = list(dict.fromkeys([e["target"], sia_data.SUCCESS, sia_data.SILENCE] + [h for h in e["hard"] if h != e["target"]]))
                    z = hc(encode_text(cands)) @ hs(o.hidden_states[a.layer][0, pos])
                    ok.append((int(z.argmax()) == 0, e["kind"]))
            print("sia eval:", s + 1, round(float(np.mean([x for x, _ in ok])), 3),
                  {k: round(float(np.mean([x for x, kk in ok if kk == k])), 2) for k in sorted({k for _, k in ok})}, flush=True)
            model.train(); hs.train(); hc.train()
        if (s + 1) % a.save_every == 0 or s == steps - 1:
            lora = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save({"lora": lora, "hs": hs.state_dict(), "hc": hc.state_dict(), "speak": speak.state_dict(), "protos": protos.data, "t": logit_t.data, "b": logit_b.data,
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s + 1,
                        "rng": {"order": order_rng.getstate(), "order_list": list(order), "mode": mode_rng.getstate(),
                                "py": random.getstate(), "pair": pair_rng.getstate()}}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            model.save_pretrained(os.path.join(a.save_dir, "adapter"))
            if str(s + 1) in a.snapshot_steps.split(","):
                model.save_pretrained(os.path.join(a.save_dir, f"adapter_s{s + 1}"))
    json.dump(vars(a), open(os.path.join(a.save_dir, "args.json"), "w"), indent=1)
    print("sia: done", flush=True)


if __name__ == "__main__":
    main()
