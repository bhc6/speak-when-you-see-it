"""Supplement G: success and failure cases of ECT (seed 0, greedy, QICD test set), one per outcome class, chosen by the fixed
rule of analysis_code/qual_examples/select_examples.py (lower median of annotated step duration within each class).
Frames: 4K GoPro frames at the model's 2 fps frame times (analysis_code/qual_examples/extract_frames.py; originals in
data/vast_qic/paper_assets/qual_4k/, 1280-px copies in fig/qual/). Every text in the figure is read from selected.json.
Run: python3 make_supp_qual.py  -> fig/supp_qual.pdf"""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT = "DATA_ROOT"; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/fig"; Q = f"{OUT}/qual"
SRC = f"{ROOT}/paper_assets/qual_4k"; SELJ = f"{ROOT}/analysis_code/qual_examples/selected.json"
ORANGE, INK, INK2 = "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
os.makedirs(Q, exist_ok=True); D = json.load(open(SELJ)); M = json.load(open(f"{SRC}/meta.json")); S = D["selected"]
shutil.copy(SELJ, f"{Q}/selected.json"); shutil.copy(f"{SRC}/meta.json", f"{Q}/meta.json")
for c in M:  # 1280-px copies for the paper source
    os.makedirs(f"{Q}/{M[c]['dir']}", exist_ok=True)
    for f, _, _ in M[c]["frames"]:
        dst = f"{Q}/{M[c]['dir']}/{f}"
        if not os.path.exists(dst): Image.open(f"{SRC}/{M[c]['dir']}/{f}").convert("RGB").resize((1280, 720), Image.LANCZOS).save(dst, quality=90)
NAME = {"A": "On-time confirmation", "B": "Timely alarm", "C": "False alarm", "D": "Missed mistake", "E": "Mistimed confirmation"}
SHORT = {"A": "confirms on time", "B": "alarms on time", "C": "false alarm", "D": "misses the mistake", "E": "confirms off time", None: "step not aligned by the official parser"}
def clip(s, n): s = " ".join(s.split()); return s if len(s) <= n else s[:n - 1].rstrip() + "…"
def rel(t, o): return f"{t - o['start']:+.1f} s"
def says(o):
    p = [f"“{clip(f['text'].replace('Feedback:', '').strip(), 70)}” at {rel(f['t'], o)}" for f in o["feedback"][:1]]
    if len(o["feedback"]) > 1: p[-1] += f" (+{len(o['feedback']) - 1} more)"
    p += [f"Success at {rel(t, o)}" for t in o["success"][:1]]
    return "; ".join(p) if p else "nothing"
fig = plt.figure(figsize=(7.0, 6.3)); rows = "ABCDE"; fw, fh, top, rh = 1.0 / 6, 0.098, 0.985, 0.197
for r, c in enumerate(rows):
    o = S[c]; m = M[c]; y0 = top - r * rh
    fig.text(0.0, y0, f"{c}  {NAME[c]}", fontsize=7.5, weight="bold", color=INK, va="top")
    fig.text(0.0, y0 - .017, f"{clip(o['instruction'], 72)}  (recording {o['video']}, step {o['step'] + 1}; annotated duration {o['dur']:.1f} s; "
             f"{o['n_candidates']} steps in this class)", fontsize=6.3, color=INK2, va="top")
    for k, (f, _, tr) in enumerate(m["frames"]):
        ax = fig.add_axes([k * fw + .002, y0 - .03 - fh, fw - .004, fh]); ax.imshow(np.asarray(Image.open(f"{Q}/{m['dir']}/{f}"))); ax.set_axis_off()
        ax.text(.03, .05, f"{tr:+.1f} s", transform=ax.transAxes, fontsize=5.5, color="white", va="bottom",
                bbox=dict(boxstyle="square,pad=0.15", fc=(0, 0, 0, .55), ec="none"))
    gt = "; ".join(f"“{clip(g['text'], 80)}” at {rel(g['t'], o)}" for g in o["gt_mistakes"][:1]) if o["gt_mistakes"] else f"no mistake; step ends at {rel(o['end'], o)}"
    fig.text(0.0, y0 - .136, f"Annotation: {gt}", fontsize=6.3, color=INK, va="top")
    fig.text(0.0, y0 - .152, f"ECT: {says(o)}", fontsize=6.3, color=ORANGE, va="top")
    nn = o["no_neg"]; nd = dict(o, feedback=nn["feedback"], success=nn["success"])
    fig.text(0.0, y0 - .168, f"Without paired negatives: {SHORT[nn['cls']]} — {says(nd)}", fontsize=6.3, color=INK2, va="top")
fig.savefig(f"{OUT}/supp_qual.pdf"); fig.savefig(f"{OUT}/supp_qual.png", dpi=170); plt.close(fig)
print("counts", D["counts"])
