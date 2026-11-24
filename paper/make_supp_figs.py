"""Figures of the supplementary material (same palette as the paper's figures).
S1 threshold sweeps (analysis_code/opcurve/*.json) against the frontier, with ECT's greedy points;
S2 the one mistake pair that passes the strict rule (fig/cand_mistake/, candidate 20_48);
S3 two further swap candidates (fig/hr/, s2 and s0).
Run: python3 make_supp_figs.py  -> fig/supp_sweep.pdf, fig/supp_mistake_pair.pdf, fig/supp_swap_more.pdf"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
ROOT = "DATA_ROOT"; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/fig"
sys.path.insert(0, f"{ROOT}/analysis_code"); import frontier_canon as F
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID, REGION = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.linewidth": .5, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": .5,
                     "ytick.major.width": .5, "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "legend.frameon": False})
def img(path, width=560):
    im = Image.open(path).convert("RGB"); return np.asarray(im.resize((width, round(width * im.size[1] / im.size[0])), Image.LANCZOS))

# ---------------------------------------------------------------- S1: threshold sweeps
VAR = [("+ clean", "ABL_noND", AQUA, "-"), ("+ notdone", "ABL_noCL", VIOLET, "-"), ("No paired neg.", "ABL_none", INK2, "-"), ("No neg., re-balanced", "T1v2_2400", INK2, (0, (3, 2)))]
ECTG = {"qicd": [(41.2, .217), (39.9, .284), (39.5, .267), (42.4, .221)], "mc": [(67.0, .208), (54.2, .228), (59.1, .278), (55.7, .271)]}   # greedy points, RESULTS_AUTHORITATIVE
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5)); fig.subplots_adjust(left=.07, right=.99, top=.86, bottom=.16, wspace=.18)
for ax, ds, suf, title in ((axs[0], "qicd", "test36", "QICD test set"), (axs[1], "mc", "mc", "Ego-MC-Bench")):
    h = F.H[ds]; xs = [0] + [a for a, _ in h]; ys = [h[0][1] if ds == "qicd" else .373] + [b for _, b in h]
    hh = [(a, b) for a, b in h if not (a == 0 and b < .05)]; ax.fill_between([0] + [a for a, _ in hh], [hh[0][1]] + [b for _, b in hh], color=REGION, alpha=.75, lw=0); ax.plot([0] + [a for a, _ in hh], [hh[0][1]] + [b for _, b in hh], color=INK2, lw=.7)
    for name, k, col, ls in VAR:
        c = json.load(open(f"{ROOT}/analysis_code/opcurve/jobs__{k}_prob_{suf}.json"))["curve"]
        ax.plot([p["ic"] for p in c], [p["f1"] for p in c], color=col, lw=1.2, ls=ls, marker="o", ms=2, label=name)
    if ds == "mc":
        for i, f in enumerate(("final__sft_prob_mc", "seeds__final__s1_prob_mc", "seeds__final__s2_prob_mc", "jobs__s3_prob_mc")):
            c = json.load(open(f"{ROOT}/analysis_code/opcurve/{f}.json"))["curve"]; ax.plot([p["ic"] for p in c], [p["f1"] for p in c], color=ORANGE, lw=1.0, alpha=.8, label="ECT, thresholds swept (4 seeds)" if i == 0 else None)
    ax.scatter(*zip(*ECTG[ds]), s=26, color=ORANGE, edgecolor="white", lw=.5, zorder=5, label="ECT, greedy (4 seeds)")
    ax.set_xlim(-2, 72); ax.set_ylim(-.015, .42); ax.set_xlabel("completion accuracy, IC-Acc (%)"); ax.set_title(title, fontsize=7.5, loc="left", pad=3, color=INK); ax.grid(color=GRID, lw=.4); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("mistake F1"); axs[0].text(8, .06, "video-blind region", fontsize=6, color=INK2, style="italic")
h, l = axs[1].get_legend_handles_labels(); fig.legend(h, l, fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(.53, 1.0), handlelength=1.6, columnspacing=1.1)
fig.savefig(f"{OUT}/supp_sweep.pdf"); fig.savefig(f"{OUT}/supp_sweep.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- S2: the mistake pair that passes the strict rule
cm = next(c for c in json.load(open(f"{OUT}/cand_mistake/meta.json")) if c["id"] == "20_48_E20.5"); D = f"{OUT}/cand_mistake/{cm['id']}"; p = cm["p"]
ect = [sum(p[f"T1v4_s{s}"][i] for s in range(4)) / 4 for i in (0, 1)]; non = [sum(p[k][i] for k in ("ABL_none", "ABL_none_s1", "T1v2")) / 3 for i in (0, 1)]
W, H = 7.0, 1.8; fig = plt.figure(figsize=(W, H)); X = lambda v: v / W; Y = lambda v: v / H
fw = 1.1; fh = fw * 9 / 16; SEL = (4, 6, 7, 8, 9)
for r, (pref, lab, instr) in enumerate((("mistake", "Step with a mistake", cm["pos_instr"]), ("clean", "Mistake-free step", cm["neg_instr"]))):
    ytop = H - .2 - r * (fh + .32)
    fig.text(X(.02), Y(ytop + .05), lab, fontsize=6.8, weight="bold", color=INK, va="bottom"); fig.text(X(1.25), Y(ytop + .05), "“" + instr.replace("Now ", "").rstrip(".").capitalize() + ".”", fontsize=6.4, style="italic", color=INK, va="bottom")
    for k, j in enumerate(SEL):
        ax = fig.add_axes([X(.02 + k * (fw + .02)), Y(ytop - fh), X(fw), Y(fh)]); ax.imshow(img(f"{D}/{pref}{j}.jpg"), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    for i, (name, v, col) in enumerate((("zero-shot", p["zero-shot"][r], BLUE), ("no neg.", non[r], INK2), ("ECT", ect[r], ORANGE))):
        x = 5.72 + i * .44
        if r == 0: fig.text(X(x + .17), Y(H - .12), name, fontsize=6, ha="center", color=col, weight="bold" if i == 2 else "normal")
        fig.patches.append(plt.Rectangle((X(x), Y(ytop - fh / 2 - .11)), X(.34), Y(.22), transform=fig.transFigure, fc="#fdefe8" if i == 2 else "white", ec=col if i == 2 else "#9a9a9a", lw=.8 if i == 2 else .5))
        fig.text(X(x + .17), Y(ytop - fh / 2), f"{v:.2f}", fontsize=7, ha="center", va="center", color=INK, weight="bold" if i == 2 else "normal")
fig.savefig(f"{OUT}/supp_mistake_pair.pdf"); fig.savefig(f"{OUT}/supp_mistake_pair.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- S3: two further swap candidates
hr = {s["id"]: s for s in json.load(open(f"{OUT}/hr/meta.json"))["swap"]}; PICK = ["s2_16_44_st13", "s0_22_40_st3"]
W, H = 7.0, 1.95; fig = plt.figure(figsize=(W, H)); X = lambda v: v / W; Y = lambda v: v / H; fw = .79; fh = fw * 9 / 16
for c, sid in enumerate(PICK):
    s = hr[sid]; x0 = .2 + c * 3.5; late = "never" if s.get("swap_confirm") is None else f"{s['swap_confirm'] - s['end']:+.0f} s"
    fig.text(X(x0), Y(H - .07), "“" + s["instruction"].replace("Now ", "").rstrip(".").capitalize() + ".”", fontsize=6.4, style="italic", color=INK, va="top")
    for r, (pref, lab, note) in enumerate((("orig", "original", f"confirmed {s['orig_confirm'] - s['end']:+.0f} s from the step end"), ("swap", "swapped", f"confirmed {late} from the step end"))):
        ytop = H - .34 - r * (fh + .34)
        fig.text(X(x0 - .06), Y(ytop - fh / 2), lab, fontsize=5.8, color=INK2, ha="right", va="center", rotation=90); fig.text(X(x0), Y(ytop - fh - .05), note, fontsize=5.8, color=INK, va="top")
        for k, j in enumerate((1, 2, 3, 4)):
            ax = fig.add_axes([X(x0 + k * (fw + .015)), Y(ytop - fh), X(fw), Y(fh)]); ax.imshow(img(f"{OUT}/hr/{sid}/{pref}{j}.jpg", 420), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
fig.savefig(f"{OUT}/supp_swap_more.pdf"); fig.savefig(f"{OUT}/supp_swap_more.png", dpi=200); plt.close(fig)
print("ok")
