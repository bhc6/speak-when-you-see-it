"""SIA scoring at inference (S2): the trained situation head hs, candidate head hc, type prototypes and temperature from
an sia_train.py checkpoint (last.pt, same step as the merged adapter). For an official prompt + video window:
  type probs  softmax(exp(t) * <protos, hs(h_L[last prompt token])>) over (silence, success, mistake)
  content     argmax over the TRAIN feedback bank of <hc(reply-encoding of "YES. <fb>"), hs(h)>  (retrieved feedback)
Exactly the quantities trained in sia_train.py (v2 two-level SIA, encode_reply). Bank: json list from the train split only."""
import json
import torch
import torch.nn.functional as F
import sia_train as T


def _lm(model):
    """text decoder of a merged Qwen3VLForConditionalGeneration (PreTrainedModel.base_model would return the inner
    Qwen3VLModel, so sia_train.lm_layers cannot be used here)."""
    return model.model.language_model


class SIAScorer:
    def __init__(self, model, proc, ckpt, bank_json, layer=21, linear=True):
        st = torch.load(ckpt, map_location="cuda", weights_only=False)
        self.hs = T.Head(linear=linear).cuda().eval(); self.hs.load_state_dict(st["hs"])
        self.hc = T.Head(linear=linear).cuda().eval(); self.hc.load_state_dict(st["hc"])
        self.protos = F.normalize(st["protos"].cuda().float(), dim=-1); self.t = float(st["t"].exp())
        self.model, self.proc, self.tok, self.layer = model, proc, proc.tokenizer, layer
        self.bank = json.load(open(bank_json))
        self.bank_g = torch.cat([self.hc(self._reply(self.bank[i:i + 32])) for i in range(0, len(self.bank), 32)])
        print(f"sia_score: step {st['step']} bank {len(self.bank)} layer {layer}", flush=True)

    @torch.no_grad()
    def _reply(self, texts):   # == sia_train.encode_reply
        tok, proc = self.tok, self.proc
        convs = [proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "."}]}], tokenize=False,
                                          add_generation_prompt=True) for _ in texts]
        enc = tok([c + f"YES. {t}" for c, t in zip(convs, texts)], return_tensors="pt", padding=True, padding_side="right").to("cuda")
        plen = len(tok(convs[0])["input_ids"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = _lm(self.model)(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer].float(); m = enc.attention_mask.clone(); m[:, :plen] = 0
        return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)

    @torch.no_grad()
    def score(self, hidden_states):
        """hidden_states: model output.hidden_states of a forward over the official prompt (generation prompt at end)."""
        g = self.hs(hidden_states[self.layer][0, -1])
        p = torch.softmax(self.t * (self.protos @ g), -1)
        sims = self.bank_g @ g; k = int(sims.argmax())
        return {"p_silence": float(p[0]), "p_success": float(p[1]), "p_mistake": float(p[2]), "fb": self.bank[k], "fb_sim": float(sims[k])}
