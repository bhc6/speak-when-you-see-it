"""4K GoPro frames for the selected qualitative examples (CaptainCook4D gopro_4k, HTTP seek with ffmpeg; same method and
model-window rule as the high-resolution figure frames of the paper: 6 uniform frames of the model window ending at the decision time;
frame i of the 2 fps stream <-> t = i/2 s). python extract_frames.py SELECTED_JSON OUT_DIR"""
import sys, os, json, subprocess, concurrent.futures as cf
SEL = json.load(open(sys.argv[1]))["selected"]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "download_links.json")))
def window(st, t):
    i0 = int(st * 2); seek = max(int(t * 2), i0 + 10); b0 = max(seek - 240, i0)
    return [round(b0 + (seek - 1 - b0) * k / 5) for k in range(6)]
tasks, meta = [], {}
for c, o in SEL.items():
    idx = window(o["start"], o["decision_t"]); d = f"{OUT}/{c}_{o['video']}_st{o['step']}"
    meta[c] = dict(dir=os.path.basename(d), video=o["video"], step=o["step"], decision_t=o["decision_t"],
                   frames=[(f"f{k}.jpg", round(i / 2, 1), round(i / 2 - o["start"], 1)) for k, i in enumerate(idx)])
    tasks += [(o["video"], i / 2, f"{d}/f{k}.jpg") for k, i in enumerate(idx)]
def run(t):
    v, ts, out = t; os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): return 0
    return subprocess.call(["timeout", "300", "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ts:.3f}", "-i", L[v]["gopro_4k"], "-frames:v", "1", "-q:v", "2", out])
with cf.ThreadPoolExecutor(4) as ex: rc = list(ex.map(run, tasks))
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1); print("tasks", len(tasks), "failed", sum(r != 0 for r in rc))
