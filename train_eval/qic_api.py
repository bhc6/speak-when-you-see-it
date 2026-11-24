"""Official per-step protocol with a CLOSED/API model (OpenAI-compatible chat API: OpenRouter or DashScope).
Unchanged: dataset, GT restarts, 5-s checks, window [step start, t], official prompts (system + user), 'yes' parsing.
Changed (disclosed): the window is uniformly subsampled to QIC_API_FRAMES frames (default 16, last frame kept) and sent
as JPEG images (QIC_API_WIDTH px wide, default 448) in temporal order with their timestamps; temperature 0.
Every request/response is cached in QIC_API_CACHE (jsonl, keyed by video/step/check/mode) -> resumable, no double charge.
Keys are read from the environment only (OPENROUTER_API_KEY or DASHSCOPE_API_KEY); never logged.
    QIC_API_MODEL=google/gemini-2.5-pro QIC_API_BASE=https://openrouter.ai/api/v1 python qic_api.py <baseline args>"""
from __future__ import annotations
import base64, io, json, os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import qic_shard  # noqa: E402
import vlm_common as V  # noqa: E402
from PIL import Image  # noqa: E402
baseline = qic_shard.baseline
MODEL = os.environ["QIC_API_MODEL"]; BASE = os.environ.get("QIC_API_BASE", "https://openrouter.ai/api/v1")
if BASE == "us": BASE, KEY = os.environ["DASHSCOPE_US_BASE_URL"].rstrip("/"), os.environ.get("DASHSCOPE_US_API_KEY")
elif BASE == "cn": BASE, KEY = "https://ENDPOINT/compatible-mode/v1", os.environ.get("DASHSCOPE_API_KEY")
else: KEY = os.environ.get("OPENROUTER_API_KEY") if "openrouter" in BASE else os.environ.get("DASHSCOPE_API_KEY")
NF = int(os.environ.get("QIC_API_FRAMES", "16")); W = int(os.environ.get("QIC_API_WIDTH", "448"))
CACHE = os.environ["QIC_API_CACHE"]; _cache = {}
FMT = os.environ.get("QIC_API_FMT") == "1"
import glob  # noqa: E402
for _f in sorted(glob.glob(os.path.join(os.path.dirname(CACHE) or ".", "cache*.jsonl"))):   # all shards' caches: re-sharding never pays twice
    for l in open(_f):
        try: r = json.loads(l); _cache[r["k"]] = r["out"]
        except json.JSONDecodeError: pass   # a line cut by a killed process
STAT = {"calls": 0, "cached": 0, "in_tok": 0, "out_tok": 0}

def _jpeg(f):
    im = Image.fromarray(f); im = im.resize((W, int(W * im.height / im.width))); b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def _post(payload):
    import urllib.request
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for i in range(12):   # ~45 min of back-off: survives account/quota interruptions (e.g. balance top-up)
        try:
            with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            info = ""
            if hasattr(e, "code"):
                try: info = f" {e.code} {e.read()[:160]!r}"
                except Exception: info = f" {e.code}"  # noqa: BLE001
                if "DataInspectionFailed" in info or "data_inspection_failed" in info:
                    print("qic_api: provider content filter refused this request -> answered NO (counted, disclosed)", flush=True)
                    return {"choices": [{"message": {"content": "NO"}}], "usage": {}, "model": MODEL, "filtered": True}
            print(f"qic_api: retry {i} {type(e).__name__}{info}", flush=True); time.sleep(min(300, 5 * 2 ** i))
    raise RuntimeError("API failed 12 times")

def _ask(mode, model, processor, video=None, instruction=None, max_new_tokens=128):
    end = (video.__array_interface__["data"][0] - video.base.__array_interface__["data"][0]) // video.base.strides[0] + len(video)
    fp = hashlib.md5(video[-1].tobytes()[::4093] + video[0].tobytes()[::4093]).hexdigest()   # identifies the video content
    msgs = baseline.build_messages(mode, instruction=instruction)   # (memory prompt included when QIC_API_MEM=1)
    ph = hashlib.md5((msgs[0]["content"] if isinstance(msgs[0]["content"], str) else json.dumps(msgs[0]["content"])).encode() + msgs[1]["content"][1]["text"].encode()).hexdigest()
    k = hashlib.md5(f"{MODEL}|{NF}|{W}|{mode}|{instruction}|{end}|{len(video)}|{fp}|{ph}".encode()).hexdigest()
    if k in _cache: STAT["cached"] += 1; return _cache[k]
    clip = V.subsample(video, NF); n = len(video)
    idx = np.unique(np.linspace(0, n - 1, len(clip)).round().astype(int)) if n > NF else np.arange(n)
    parts = [{"type": "text", "text": f"The video below has {len(clip)} frames sampled in temporal order from the last {n / 2:.0f} s; the last frame is the current moment."}]
    for t, f in zip(idx, clip): parts += [{"type": "text", "text": f"[{t / 2:.1f} s]"}, {"type": "image_url", "image_url": {"url": _jpeg(f)}}]
    q = msgs[1]["content"][1]["text"]
    if FMT: q += " Start your answer with YES or NO, then give at most one sentence."   # API models only (disclosed): long free-form answers get cut at max_tokens
    parts.append({"type": "text", "text": q})
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_new_tokens, "messages": [{"role": "system", "content": msgs[0]["content"]}, {"role": "user", "content": parts}]}
    if MODEL.startswith("qwen3"): body["enable_thinking"] = False   # non-thinking mode (disclosed); thinking would consume max_tokens
    r = _post(body)
    out = (r["choices"][0]["message"].get("content") or "").strip(); u = r.get("usage", {})
    STAT["calls"] += 1; STAT["in_tok"] += u.get("prompt_tokens", 0); STAT["out_tok"] += u.get("completion_tokens", 0)
    with open(CACHE, "a") as fh: fh.write(json.dumps({"k": k, "mode": mode, "out": out, "usage": u, "model": r.get("model"), "t": time.time(), "filtered": r.get("filtered", False)}) + "\n")
    _cache[k] = out
    if STAT["calls"] % 50 == 0: print(f"qic_api: {STAT}", flush=True)
    return out

baseline.load_model_and_processor = lambda *a, **k: (None, None)
baseline.get_qwen_vl_output = _ask
if os.environ.get("QIC_API_MEM") == "1":   # R1 protocol: 1-min feedback memory prompt + yes-gate of qic_online wraps the API call
    os.environ["QIC_MEM"] = "1"
    if FMT: os.environ["QIC_STRICT"] = "1"   # answers start with YES/NO -> parse the leading token only
    import qic_online  # noqa: E402,F401
    baseline.load_model_and_processor = lambda *a, **k: (None, None)   # qic_greedy (imported by qic_online) would load a local model
print(f"qic_api: model={MODEL} base={BASE.split(chr(47))[2]} frames={NF} width={W} mem={os.environ.get('QIC_API_MEM')} fmt={FMT} key_present={bool(KEY)}", flush=True)
if __name__ == "__main__":
    baseline.run(baseline.parse_args()); print(f"qic_api: final {STAT}", flush=True)
