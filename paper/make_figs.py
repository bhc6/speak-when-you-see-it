"""Figures of the paper (Fig. 1 teaser, Fig. 2 QICD frontier, Fig. 3 decision decomposition).
Numbers are read from audit_1114/ (qwen_api_results.json, decision_decomp*.json), refcheck/published_frontier_egomc.json and
analysis_code/frontier_canon.py; the ECT / zero-shot points are the rows of audit_1114/RESULTS_AUTHORITATIVE.md named in the comments.
Run:  python3 make_figs.py   (writes fig/fig1_teaser.pdf, fig2_frontier_qicd.pdf, fig3_decomp.pdf, fig4_pairs.pdf)
Palette: validated with the dataviz validator (blue / orange / aqua / yellow / violet, light surface); identity is never colour-alone
(marker shape + direct labels)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = "DATA_ROOT"
sys.path.insert(0, f"{ROOT}/analysis_code"); import frontier_canon as F
A = f"{ROOT}/audit_1114"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID, REGION = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.linewidth": .5, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.labelcolor": INK,
                     "pdf.fonttype": 42, "legend.frameon": False})

Q = {r["run"]: r for r in json.load(open(f"{A}/qwen_api_results.json"))}
API = ["q38max", "q38omniflash", "q35_397a17", "q35_122a10", "q35_35a3", "q35_27b"]
# RESULTS_AUTHORITATIVE rows: sft_greedy / s1_greedy / s2_greedy / s3_greedy
ECT = {"qicd": [(41.2, .217), (39.9, .284), (39.5, .267), (42.4, .221)], "mc": [(67.0, .208), (54.2, .228), (59.1, .278), (55.7, .271)]}
# C2 (InternVL3.5-2B), C6 (LLaVA-OV-7B)
XB = {"qicd": [(47.9, .256), (49.6, .171)], "mc": [(64.0, .104), (64.0, .090)]}
# R0, R1, C1 InternVL3.5-2B, C4 InternVL3.5-8B, C5 LLaVA-OV-7B, q25_mem Qwen2.5-VL-7B (+ qicd_8b_mem Qwen3-VL-8B on QICD)
ZS = {"qicd": [(4.2, .074), (10.5, .146), (32.8, 0), (3.4, 0), (0, 0), (1.7, 0), (2.9, 0)], "mc": [(14.0, .147), (14.4, .215), (49.2, 0), (11.4, 0), (0, 0), (8.0, 0)]}
PUB = [(r[1], r[2]) for r in json.load(open(f"{ROOT}/refcheck/published_frontier_egomc.json")) if r[1] is not None]  # all 18 published per-step results (SI Tab.2 + Tab.3)


def frontier_panel(ax, ds, published=False, legend=True):
    h = F.H[ds]; xs = [0] + [a for a, _ in h]; ys = [h[0][1]] + [b for _, b in h]
    ax.fill_between(xs, ys, color=REGION, alpha=.75, lw=0, zorder=1); ax.plot(xs, ys, color=INK2, lw=.8, zorder=2)
    ax.scatter(*zip(*ZS[ds]), s=13, marker="s", color=BLUE, edgecolor="white", lw=.4, zorder=4, label="zero-shot, 2B–8B")
    api = [(Q[f"API_{k}_{'test36' if ds == 'qicd' else 'mc'}"]["ic"], Q[f"API_{k}_{'test36' if ds == 'qicd' else 'mc'}"]["F1"]) for k in API]
    ax.scatter(*zip(*api), s=20, marker="P", color=AQUA, edgecolor="white", lw=.4, zorder=4, label="zero-shot, 27B and larger")
    if published: ax.scatter(*zip(*PUB), s=17, marker="D", color=VIOLET, edgecolor="white", lw=.4, zorder=4, label="published")
    ax.scatter(*zip(*XB[ds]), s=20, marker="^", color=ORANGE, edgecolor="white", lw=.4, zorder=5, label="ECT, other backbones")
    ax.scatter(*zip(*ECT[ds]), s=30, marker="o", color=ORANGE, edgecolor="white", lw=.5, zorder=6, label="ECT (ours), 4 seeds")
    ax.set_xlim(-2, 72); ax.set_ylim(-.015, .42); ax.set_xlabel("completion accuracy, IC-Acc (%)"); ax.set_ylabel("mistake F1")
    ax.grid(color=GRID, lw=.4, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if legend: ax.legend(fontsize=5.6, loc="upper right", handletextpad=.2, borderaxespad=.1, labelspacing=.25)
    return api


# ---------------------------------------------------------------- Fig. 1: (a) time-matched pair, (b) Ego-MC frontier
fig = plt.figure(figsize=(7.0, 2.7))
a = fig.add_axes([.005, .02, .49, .93]); a.set_xlim(0, 100); a.set_ylim(0, 100); a.axis("off")
a.text(0, 97, "(a) Same clock, different evidence", fontsize=8, weight="bold", color=INK, va="top")
a.text(0, 87, "Two steps of one recording", fontsize=6.3, color=INK2, va="top")
a.text(79, 87, "\"Is the step complete?\"", fontsize=6.3, color=INK, ha="center", va="top", style="italic")
x0, x1, xe = 12, 56, 43          # timeline start, end of drawing area, position of E
for y, lab, sub, done in ((68, "Step A", "ends at E", True), (48, "Step B", "20 s or more to go", False)):
    a.text(0, y, lab, fontsize=7, weight="bold", color=INK, va="center"); a.text(0, y - 7, sub, fontsize=5.6, color=INK2, va="center")
    a.add_patch(FancyBboxPatch((x0, y - 3.2), xe - x0, 6.4, boxstyle="round,pad=0,rounding_size=1.2", fc=BLUE if done else "#9ec5f4", ec="none"))
    if not done:
        a.add_patch(FancyBboxPatch((xe + .6, y - 3.2), x1 - xe - .6, 6.4, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#9ec5f4", lw=.7, ls=(0, (2, 1.5))))
    a.text(x0 + 1.5, y, "evidence: step finished" if done else "evidence: still in progress", fontsize=5.4, color="white" if done else INK, va="center")
a.plot([xe, xe], [40, 76], color=INK, lw=.7, ls=(0, (3, 2))); a.text(xe, 78.5, "E", fontsize=7, weight="bold", ha="center", color=INK)
a.annotate("", xy=(xe, 37.5), xytext=(x0, 37.5), arrowprops=dict(arrowstyle="<->", color=INK2, lw=.5)); a.text((x0 + xe) / 2, 34, "same recording, same elapsed time E", fontsize=5.4, color=INK2, ha="center", va="top")
cx = (70, 88)
for x, t in zip(cx, ("Step A", "Step B")): a.text(x, 77.5, t, fontsize=6, color=INK2, ha="center")
rows = ((68, "clock-only rule", ("Yes", "Yes"), (INK2, INK2), "same input, same answer"), (48, "ECT", ("Yes", "No, wait"), (ORANGE, ORANGE), "answers from the evidence"))
for y, name, ans, cols, note in rows:
    a.text(62.5, y + 6.3, name, fontsize=5.8, color=INK, weight="bold", ha="left", va="center")
    for x, t in zip(cx, ans):
        a.add_patch(FancyBboxPatch((x - 7.5, y - 3.4), 15, 6.8, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec=cols[0], lw=.9))
        a.text(x, y, t, fontsize=6.3, ha="center", va="center", color=INK)
    a.text(79, y - 7.2, note, fontsize=5.2, color=INK2, ha="center", va="center")
a.plot([0, 98], [24, 24], color=GRID, lw=.6)
a.text(0, 19, "On 931 such test pairs, picking the right side (completion / mistake question):", fontsize=6, color=INK, va="center")
for x, name, v, c, w in ((0, "clock-only, by construction", "0.50 / 0.50", INK2, "normal"), (37, "zero-shot", "0.66 / 0.59", BLUE, "normal"), (62, "ECT", "0.90–0.92 / 0.75–0.77", ORANGE, "bold")):
    a.scatter([x + 1.2], [9], s=16, color=c, marker="o"); a.text(x + 3.5, 12, name, fontsize=5.6, color=INK2, va="center"); a.text(x + 3.5, 5.5, v, fontsize=6.6, color=INK, va="center", weight=w)

b = fig.add_axes([.565, .16, .425, .60])
frontier_panel(b, "mc", published=True, legend=False)
fig.text(.515, .95, "(b) Reachable without video (Ego-MC-Bench)", fontsize=8, weight="bold", color=INK, va="top")
b.legend(fontsize=5.6, ncol=3, loc="lower left", bbox_to_anchor=(-.13, 1.0), handletextpad=.2, columnspacing=.9, labelspacing=.3, borderaxespad=.2)
b.scatter([0], [.373], s=26, marker="X", color=INK, zorder=7, clip_on=False)
b.annotate("never looks at the video:\nalarm at 25 s, confirm at 30 s", (0, .373), xytext=(8, .385), fontsize=5.4, color=INK, va="center")
b.text(16, .075, "video-blind\nregion", fontsize=6, color=INK2, style="italic", linespacing=1.0, zorder=6)
b.annotate("best published", (37.1, .20), xytext=(20.5, .262), fontsize=5.4, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=.4, shrinkA=0, shrinkB=2))
b.annotate("ECT, 2B", (59.1, .278), xytext=(60.5, .305), fontsize=6, color=INK, weight="bold")
fig.savefig(f"{OUT}/fig1_teaser.pdf"); fig.savefig(f"{OUT}/fig1_teaser.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: video-blind frontier, both benchmarks (full width, full points)
fig, (axq, axm) = plt.subplots(1, 2, figsize=(7.0, 2.25)); fig.subplots_adjust(left=.065, right=.99, top=.79, bottom=.155, wspace=.18)
api = frontier_panel(axq, "qicd", legend=False)
axq.set_title("QICD Advanced Planning, test (36 videos)", fontsize=7.2, weight="bold", loc="left", pad=3)
axq.text(9, .235, "video-blind region", fontsize=6, color=INK2, style="italic")
axq.annotate("Qwen3.8-Max", api[0], xytext=(24.5, .148), fontsize=5.4, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=.4, shrinkA=0, shrinkB=2))
axq.annotate("Qwen3-VL-2B", (10.5, .146), xytext=(1.5, .168), fontsize=5.4, color=INK)
axq.annotate("InternVL3.5-2B", (32.8, 0), xytext=(22, .018), fontsize=5.4, color=INK)
axq.annotate("ECT, 2B", (39.9, .284), xytext=(33.5, .312), fontsize=6, color=INK, weight="bold")
apim = frontier_panel(axm, "mc", published=True, legend=False)
axm.set_title("Ego-MC-Bench (39 videos, never trained on)", fontsize=7.2, weight="bold", loc="left", pad=3)
axm.scatter([0], [.373], s=26, marker="X", color=INK, zorder=7, clip_on=False)
axm.annotate("never looks at the video:\nalarm at 25 s, confirm at 30 s", (0, .373), xytext=(8, .385), fontsize=5.4, color=INK, va="center")
axm.text(16, .075, "video-blind\nregion", fontsize=6, color=INK2, style="italic", linespacing=1.0, zorder=6)   # two lines, in the empty area between the points
axm.annotate("best published", (37.1, .20), xytext=(20.5, .262), fontsize=5.4, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=.4, shrinkA=0, shrinkB=2))
axm.annotate("Qwen3.8-Max", apim[0], xytext=(40, .36), fontsize=5.4, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=.4, shrinkA=0, shrinkB=2))
axm.annotate("ECT, 2B", (59.1, .278), xytext=(60.5, .305), fontsize=6, color=INK, weight="bold")
h, l = axm.get_legend_handles_labels()
fig.legend(h, l, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(.53, 1.0), handletextpad=.25, columnspacing=1.4, frameon=False)
fig.savefig(f"{OUT}/fig2_frontier.pdf"); fig.savefig(f"{OUT}/fig2_frontier.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: decision decomposition on mistake-free steps (QICD test, Ego-MC-Bench)
DD = json.load(open(f"{A}/decision_decomp.json"))
cats = [("HIT", "confirmed in time", BLUE), ("voided by false alarm", "voided by a false alarm", ORANGE), ("confirmed too early", "confirmed too early", AQUA),
        ("confirmed too late", "confirmed too late", YELLOW)]   # "never confirmed" (< 2% for most models) is merged into the grey rest
ROWS3 = {"qicd": [("Qwen3-VL-2B, zero-shot", ["R1"]), ("Qwen3.8-Max, zero-shot", ["max"]), ("Plain fine-tuning", ["T1plain"]), ("No paired neg.", ["none_s0", "none_s1"]),
                  ("+ clean", ["noND_s0", "noND_s1"]), ("ECT", ["ECT_s0", "ECT_s1", "ECT_s2", "ECT_s3"]), ("ECT, video swapped", ["swap"])],
         "mc": [("Qwen3-VL-2B, zero-shot", ["R1"]), ("Qwen3.8-Max, zero-shot", ["max"]), ("No paired neg.", ["none_s0", "none_s1"]),
                ("+ clean", ["noND_s0", "noND_s1"]), ("ECT", ["ECT_s0", "ECT_s1", "ECT_s2", "ECT_s3"])]}
fig = plt.figure(figsize=(7.0, 1.95))
axq = fig.add_axes([.155, .19, .325, .62]); axm = fig.add_axes([.655, .19 + .62 * 2 / 7, .325, .62 * 5 / 7])
for ax, ds, title in ((axq, "qicd", "QICD test"), (axm, "mc", "Ego-MC-Bench (never trained on)")):
    D = DD[ds]; rows = ROWS3[ds]
    for i, (lab, ks) in enumerate(rows):
        left = 0
        for c, name, col in cats:
            v = sum(D[k][c] for k in ks) / len(ks)
            ax.barh(i, v, left=left, color=col, height=.68, edgecolor="white", lw=.8, label=name if i == 0 else None)
            if v >= 7: ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color="white" if col in (BLUE, VIOLET, ORANGE) else INK)
            left += v
        ax.barh(i, 100 - left, left=left, color="#ecebe7", height=.68, edgecolor="white", lw=.8, label="never confirmed or not reached" if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6, color=INK); ax.set_ylim(len(rows) - .5, -.5); ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("% of mistake-free steps"); ax.set_title(title, fontsize=6.8, pad=3, color=INK, loc="left")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
h, l = axq.get_legend_handles_labels()
fig.legend(h, l, fontsize=5.8, ncol=6, loc="upper center", bbox_to_anchor=(.5, 1.0), handlelength=1, handletextpad=.35, columnspacing=1.2)
fig.savefig(f"{OUT}/fig3_decomp.pdf"); fig.savefig(f"{OUT}/fig3_decomp.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: time-matched evidence pairs (QICD test, 931 pairs)
E = json.load(open(f"{A}/epair_summary.json"))
for _k in ("ABL_noCL_s2", "ABL_noCL_s3"): E[_k] = {"mis_pairacc": json.load(open(os.path.join(HERE if "HERE" in dir() else os.path.dirname(os.path.abspath(__file__)), "data_addendum_1116.json")))[_k]["mis_pairacc"], "done_pairacc": json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_addendum_1116.json")))[_k]["done_pairacc"]}
prow = [("Qwen3-VL-2B, zero-shot", ["R0_zs"], BLUE), ("No paired neg.", ["ABL_none", "ABL_none_s1"], INK2), ("+ notdone", ["ABL_noCL", "ABL_noCL_s1", "ABL_noCL_s2", "ABL_noCL_s3"], INK2),
        ("+ clean", ["ABL_noND", "ABL_noND_s1"], INK2), ("ECT", ["T1v4_s0", "T1v4_s1", "T1v4_s2", "T1v4_s3"], ORANGE),
        ("InternVL3.5-2B, no neg.", ["C3_ivl2b_none"], INK2), ("InternVL3.5-2B, ECT", ["C2_ivl2b"], ORANGE), ("LLaVA-OV-7B, ECT", ["C6_llava7b"], ORANGE)]
fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.05), sharey=True); fig.subplots_adjust(left=.35, right=.975, top=.9, bottom=.17, wspace=.12)
for ax, tag, title in ((axs[0], "done", "completion pairs"), (axs[1], "mis", "mistake pairs")):
    for i, (lab, ks, col) in enumerate(prow):
        vs = [E[k][f"{tag}_pairacc"] for k in ks]; v = sum(vs) / len(vs)
        ax.barh(i, v - .5, left=.5, color=col, height=.62, alpha=.95 if col != INK2 else .55, lw=0)
        if len(vs) > 1: ax.scatter(vs, [i] * len(vs), s=5, color="white", edgecolor=INK, lw=.4, zorder=4)
        ax.text(max(vs) + .012, i, f"{v:.2f}", va="center", fontsize=5.6, color=INK)
    ax.axvline(.5, color=INK, lw=.7); ax.set_xlim(.48, 1.02); ax.set_xticks([.5, .6, .7, .8, .9, 1.0]); ax.set_xticklabels(["0.5", "", "0.7", "", "0.9", ""])
    ax.set_title(title, fontsize=6.5, pad=3, color=INK); ax.grid(axis="x", color=GRID, lw=.4); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.axhline(4.5, color=GRID, lw=.6)
axs[0].set_yticks(range(len(prow))); axs[0].set_yticklabels([r[0] for r in prow], fontsize=6, color=INK); axs[0].invert_yaxis()
fig.text(.66, .035, "pair accuracy  (0.5 = any clock-only policy)", ha="center", fontsize=6.2, color=INK)
fig.savefig(f"{OUT}/fig4_pairs.pdf"); fig.savefig(f"{OUT}/fig4_pairs.png", dpi=220); plt.close(fig)
print("ok")
