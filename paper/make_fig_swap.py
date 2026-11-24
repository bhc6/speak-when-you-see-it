"""Fig. 5 (single column): the swap example of Sec. 6. Candidate s6 of fig/hr/README.md (rule: test36, ECT seed 0, mistake-free step
confirmed on time in the original; after swapping in another recording of the same recipe at the same frame indices, no feedback
and no confirmation within tolerance; 28 steps qualify; the 7 nearest the median step duration are candidates).
Times are from fig/hr/meta.json (start, end, orig_confirm, swap_confirm); frames are the HR copies at the model's 2 fps time stamps.
Run: python3 make_fig_swap.py  (writes fig/fig5_swap.pdf/.png)"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "fig"); HR = os.path.join(OUT, "hr")
ORANGE, BLUE, INK, INK2, INK3, GRID, REGION = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#9a9a9a", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.linewidth": .5,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2, "xtick.major.width": .5})
C = next(e for e in json.load(open(f"{HR}/meta.json"))["swap"] if e["id"] == "s6_16_20_st7")
D = os.path.join(HR, C["id"]); st, end = C["start"], C["end"]
oc, sc = C["orig_confirm"] - st, C["swap_confirm"] - st; dur = end - st
SEL = (2, 3, 4, 5)
def frame(name, width=640):
    im = Image.open(os.path.join(D, name)).convert("RGB")
    return np.asarray(im.resize((width, round(width * im.size[1] / im.size[0])), Image.LANCZOS))

W, H = 3.3, 2.06
fig = plt.figure(figsize=(W, H)); X = lambda v: v / W; Y = lambda v: v / H
fig.text(X(.03), Y(H - .05), f"Step: “{C['instruction']}”", fontsize=6.6, style="italic", color=INK, va="top")
lab_w, fw, gap = .5, .68, .02; fh = fw * 9 / 16
rows = (("original", "frames_orig", "orig", H - .25), ("swapped", "frames_swap", "swap", H - .25 - fh - .08))
for lab, key, pref, ytop in rows:
    fig.text(X(.03), Y(ytop - fh / 2), lab, fontsize=6, color=INK2, va="center", linespacing=1.0)
    for k, j in enumerate(SEL):
        name, ts = C[key][j]
        ax = fig.add_axes([X(lab_w + k * (fw + gap)), Y(ytop - fh), X(fw), Y(fh)])
        ax.imshow(frame(name), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if pref == "orig": ax.set_title(f"{max(ts, 0):.0f} s", fontsize=5.6, color=INK2, pad=1.5)
# timeline: when does ECT confirm?
t = fig.add_axes([X(lab_w), Y(.27), X(W - lab_w - .05), Y(.47)])
t.axvspan(end - st - 15, end - st + 15, color=REGION, alpha=.7, lw=0)
t.axvline(dur, color=INK2, lw=.6, ls=(0, (2, 1.5)))
for y, (x, txt, col) in ((1, (oc, "confirms in time", ORANGE)), (0, (sc, f"confirms {sc - dur:.0f} s after the step ended", ORANGE))):
    t.plot([0, x], [y, y], color=GRID, lw=2.2, solid_capstyle="butt", zorder=1)
    t.scatter([x], [y], s=22, color=col, edgecolor="white", lw=.5, zorder=3)
    t.text(x + 3 if y == 1 else x - 3, y + .32, txt, fontsize=5.8, color=INK, ha="left" if y == 1 else "right", va="center")
t.text(dur, 1.78, "step ends", fontsize=5.6, color=INK2, ha="center", va="bottom")
t.set_ylim(-.55, 1.75); t.set_xlim(-2, sc + 8); t.set_yticks([0, 1]); t.set_yticklabels(["swapped", "original"], fontsize=5.8, color=INK2)
t.tick_params(axis="y", length=0); t.set_xlabel("seconds since the step began (ECT, seed 0)", fontsize=6, labelpad=1); t.tick_params(axis="x", labelsize=5.6)
for sp in ("top", "right", "left"): t.spines[sp].set_visible(False)
fig.savefig(f"{OUT}/fig5_swap.pdf"); fig.savefig(f"{OUT}/fig5_swap.png", dpi=250); print("ok", round(oc - dur, 1), round(sc - dur, 1))
