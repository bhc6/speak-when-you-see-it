"""Fig. 1, variant v3 (11-17): v2 plus the construction of a time-matched pair, folded into the row headers (one progress bar per
step on a shared time axis with the check marked at the same elapsed time) and a "label" column. Same size as v2. Writes fig1_teaser_v3.*
Fig. 1 (full width). Story in three bands, read left to right, top to bottom:
 (a) phenomenon: two windows of one QICD test recording checked at the same elapsed time (real frames; 6 uniform frames per window)
     and what each policy answers; (b) the hard number: pair accuracy on all 931 time-matched test pairs; (c) field level:
     the video-blind frontier on Ego-MC-Bench with published results.
Restraint: everything in greys; only ECT is highlighted (one accent colour).
Pair = pre-registered rule (appendix E) + equal window length; see fig/fig1_frames/meta.json, meta3.json.
Numbers: P(yes) from evidence_pair outputs (TMP_PATH, copied below); pair accuracies from audit_1114/epair_summary.json;
frontier from analysis_code/frontier_canon.py; published points from refcheck/published_frontier_egomc.json; ECT points = RESULTS_AUTHORITATIVE rows."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Rectangle
ROOT = "DATA_ROOT"; sys.path.insert(0, f"{ROOT}/analysis_code"); import frontier_canon as F
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "fig"); FD = os.path.join(OUT, "fig1_frames")
HR = os.path.join(OUT, "hr", "c3_10_48_E20.5")   # 1280x720 frames from the 4K source, same time stamps as fig1_frames/meta3.json (see fig/hr/README.md)
from PIL import Image
import numpy as np
def frame(tag, j, width=720):
    """HR frame j of side A (done) or B (not yet), downscaled to ~560 dpi at print size; embedded without resampling (interpolation='none')."""
    im = Image.open(os.path.join(HR, f"{'A_done' if tag == 'A' else 'B_notyet'}{j}.jpg")).convert("RGB")
    return np.asarray(im.resize((width, round(width * im.size[1] / im.size[0])), Image.LANCZOS))
# same palette as make_figs.py (ECT = orange, zero-shot = blue, everything else in greys)
ACC, BLUE, INK, INK2, INK3, GRID, REGION = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#9a9a9a", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.edgecolor": INK3, "axes.linewidth": .5,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": .5, "ytick.major.width": .5})
M = json.load(open(f"{FD}/meta.json")); M3 = json.load(open(f"{FD}/meta3.json"))
ZS, ECTP = (0.28, 0.82), (0.69, 0.20)          # P(yes) step A / step B: zero-shot; ECT mean of 4 seeds
E = json.load(open(f"{ROOT}/audit_1114/epair_summary.json"))
def rng(keys, tag): v = [E[k][f"{tag}_pairacc"] for k in keys]; return min(v), max(v)
ECTK, NONE = ["T1v4_s0", "T1v4_s1", "T1v4_s2", "T1v4_s3"], ["ABL_none", "ABL_none_s1", "T1v2"]

W, H = 7.0, 2.4
fig = plt.figure(figsize=(W, H))
X = lambda v: v / W; Y = lambda v: v / H
def txt(x, y, t, **k): fig.text(X(x), Y(y), t, **k)
# ---------------- (a) frames
txt(.02, H - .06, "Same recording, same clock: both steps checked 20.5 s after they began", fontsize=7.8, weight="bold", va="top", color=INK)
fw = 1.28; fh = fw * 9 / 16; gap = .025; x0 = .02; SEL = (0, 2, 4, 5)   # 4 larger frames per step: 0, 8, 16, 20 s
rows = (("A", "“Slice with floss to make one more pinwheel.”", "step ended 5 s ago", H - .58), ("B", "“Trim the ends of the tortilla roll …”", "87 s of work still to go", H - 1.58))
BX0 = 2.98; BS = (x0 + 4 * (fw + gap) - gap - BX0) / (M["steps"]["B"]["end"] - M["steps"]["B"]["start"])   # inches per second
for tag, instr, state, ytop in rows:
    txt(x0, ytop + .06, f"Step {tag}", fontsize=7, weight="bold", color=INK, va="bottom")
    txt(x0 + .48, ytop + .06, instr, fontsize=6.6, color=INK, va="bottom", style="italic")
    # progress bar of the step on a time axis shared by both rows; the check sits at the same elapsed time E in both
    dur = M["steps"][tag]["end"] - M["steps"][tag]["start"]; Ecl = M["E"]; sx = lambda t: BX0 + t * BS; yb = ytop + .075; bh = .085
    fig.patches.append(Rectangle((X(sx(0)), Y(yb)), X(dur * BS), Y(bh), transform=fig.transFigure, fc=REGION, ec="none"))
    fig.patches.append(Rectangle((X(sx(0)), Y(yb)), X(min(dur, Ecl) * BS), Y(bh), transform=fig.transFigure, fc=INK3, ec="none"))   # elapsed part
    fig.lines.append(plt.Line2D([X(sx(Ecl))] * 2, [Y(yb - .035), Y(yb + bh + .035)], transform=fig.transFigure, color=INK, lw=.9))
    txt(sx(Ecl) + .05, yb + bh / 2, state, fontsize=6.2, color=INK2 if tag == "A" else INK, va="center")
    if tag == "A": txt(sx(Ecl), yb + bh + .05, "check at 20.5 s", fontsize=5.8, color=INK, va="bottom", ha="center")
    for k, (j, (fn, ts)) in enumerate([(j, M3[tag][j]) for j in SEL]):
        ax = fig.add_axes([X(x0 + k * (fw + gap)), Y(ytop - fh), X(fw), Y(fh)]); ax.imshow(frame(tag, j), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        cue = {("A", 2): "pinwheels on the plate", ("A", 3): "moves on to the tablet", ("B", 3): "still trimming"}.get((tag, k))
        if cue: ax.text(.975, .045, cue, transform=ax.transAxes, fontsize=5.6, color="white", ha="right", va="bottom", bbox=dict(fc="black", alpha=.55, lw=0, pad=.9))
        ax.text(.025, .045, f"{max(ts, 0):.0f} s", transform=ax.transAxes, fontsize=6, color="white", va="bottom", bbox=dict(fc="black", alpha=.55, lw=0, pad=.9))
xr = x0 + 4 * (fw + gap) + .04; cw = .40; LX = xr + .13; xr += .36   # LX: centre of the label column
txt(LX - .12, H - .22, "“Is the step complete?”", fontsize=6.8, style="italic", color=INK, va="center")
txt(LX, H - .52, "label", fontsize=6.9, ha="center", va="center", color=INK)
heads = ("clock\nonly", "zero-\nshot", "ECT\n(ours)"); HC = (INK2, BLUE, ACC)
for i, hd in enumerate(heads):
    txt(xr + i * cw + .17, H - .52, hd, fontsize=6.9, linespacing=.95, ha="center", va="center", color=HC[i], weight="bold" if i == 2 else "normal")
ans = {"A": (("yes", True), (f"{ZS[0]:.2f}", False), (f"{ECTP[0]:.2f}", True)), "B": (("yes", False), (f"{ZS[1]:.2f}", False), (f"{ECTP[1]:.2f}", True))}
for tag, _, _, ytop in rows:
    yc = ytop - fh / 2
    txt(LX, yc, "YES" if tag == "A" else "NO", fontsize=7.2, ha="center", va="center", color=INK, weight="bold")
    for i, (v, ok) in enumerate(ans[tag]):
        hl = i == 2; bx = xr + i * cw
        fig.patches.append(Rectangle((X(bx), Y(yc - .11)), X(.34), Y(.22), transform=fig.transFigure, fc="#fdefe8" if hl else "white", ec=ACC if hl else INK3, lw=.8 if hl else .5))
        txt(bx + .17, yc, v, fontsize=7.2, ha="center", va="center", color=INK, weight="bold" if hl else "normal")
for i, v in enumerate(("tie", "wrong", "right")):
    txt(xr + i * cw + .17, H - 1.58 - fh / 2 - .22, v, fontsize=6.7, ha="center", va="center", color=ACC if i == 2 else INK2, weight="bold" if i == 2 else "normal", linespacing=1.0)
# (b) pair accuracy over all 931 pairs was removed: Fig. 4 shows the same numbers in full; the caption quotes them.
fig.savefig(f"{OUT}/fig1_teaser_v3.pdf"); fig.savefig(f"{OUT}/fig1_teaser_v3.png", dpi=220); print("ok")
