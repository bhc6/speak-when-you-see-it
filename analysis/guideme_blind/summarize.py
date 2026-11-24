"""Summary numbers + figure for refcheck/J_guideme_blind.md.  .venv/bin/python summarize.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from frontier import OUT, hull_y  # noqa: E402

d = json.load(open(OUT / "behavior_frontier.json"))
seeds = d["seeds"]


def H(r, k):
    return [(h["FA_rate"], h["TPR"], h) for h in r[k]]


def qstar(r, k, target):
    """smallest mean judge quality q (0-1) on spoken anchors so that some blind hull point has CS + q*PC >= target."""
    s = r["silent_share"]
    best = 1e9
    for x, y, _ in H(r, k):
        cs, pc = s * (100 - x) / 100, (100 - s) * y / 100
        if cs >= target:
            return 0.0
        if pc > 0:
            best = min(best, (target - cs) / pc)
    return best


out = {"silent_share": seeds[0]["silent_share"], "rows": []}
for row in d["published"]:
    r = {"model": row["model"], "FA_rate": round(row["FA_rate"], 1), "TPR": round(row["TPR"], 1), "Score": row["Score"]}
    for fr in ["Q", "F"]:
        for kind in ["test", "cv"]:
            ys = [hull_y(H(s, f"hull_{fr}_{kind}"), row["FA_rate"]) for s in seeds]
            r[f"margin_{fr}_{kind}"] = [round(row["TPR"] - max(ys), 1), round(row["TPR"] - min(ys), 1)]  # >0 = outside
        r[f"qstar_F_cv"] = [round(min(qstar(s, "hull_F_cv", row["Score"]) for s in seeds), 3),
                            round(max(qstar(s, "hull_F_cv", row["Score"]) for s in seeds), 3)]
        r[f"qstar_Q_cv"] = [round(min(qstar(s, "hull_Q_cv", row["Score"]) for s in seeds), 3),
                            round(max(qstar(s, "hull_Q_cv", row["Score"]) for s in seeds), 3)]
    out["rows"].append(r)
for fr in ["Q", "F"]:
    for kind in ["test", "cv"]:
        out[f"TPR_at_FA_{fr}_{kind}"] = {x: [round(min(hull_y(H(s, f"hull_{fr}_{kind}"), x) for s in seeds), 1),
                                             round(max(hull_y(H(s, f"hull_{fr}_{kind}"), x) for s in seeds), 1)]
                                         for x in [0, 5, 10, 20, 30, 50, 70, 90]}
out["gap_counterfactual"] = d["gap_counterfactual"]
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# ---- figure: ROC plane, seed 0
r0 = seeds[0]
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.plot([0, 100], [0, 100], color="#bdbcb6", lw=1, ls=":", label="chance (random speak)")
for k, col, ls, lab in [("hull_Q_test", "#2a78d6", "-", "blind frontier Q (delay / duration prior), test-selected"),
                        ("hull_Q_cv", "#2a78d6", "--", "blind frontier Q, CV-selected"),
                        ("hull_F_test", "#eb6834", "-", "blind frontier F (+ window / hazard table), test-selected"),
                        ("hull_F_cv", "#eb6834", "--", "blind frontier F, CV-selected")]:
    xs, ys = zip(*[(h["FA_rate"], h["TPR"]) for h in r0[k]])
    ax.plot(xs, ys, color=col, lw=2, ls=ls, label=lab)
for row in d["published"]:
    ax.scatter(row["FA_rate"], row["TPR"], s=36, color="#0b0b0b", zorder=5, edgecolor="#fcfcfb", linewidth=1.5)
    off = {"Doubao-Seed-1.8": (-62, 6), "Gemini 3.1 Pro": (-20, -24), "Qwen3-VL-30B-A3B": (-30, 10),
           "Qwen3-VL-8B": (5, -12), "Qwen3-VL-8B (FT on GuideMe-Train)": (6, -14), "Qwen2.5-VL-7B": (6, -2),
           "Gemini 3 Pro": (-58, 4), "Dispider": (6, 6), "MMDuet2": (6, 16), "VideoLLM-online": (-70, -4)}
    ax.annotate(row["model"].replace(" (FT on GuideMe-Train)", " FT"), (row["FA_rate"], row["TPR"]),
                textcoords="offset points", xytext=off.get(row["model"], (4, -9)), fontsize=6.5, color="#52514e",
                arrowprops=dict(arrowstyle="-", color="#bdbcb6", lw=0.5))
ax.set_xlabel("FA rate on silent instances = FA / (FA + CS), %", color="#0b0b0b")
ax.set_ylabel("Response rate on anchors = PC / (PC + NR), %", color="#0b0b0b")
ax.set_title("GuideMe-Test, Anchor + GT history: video-blind frontier vs published models\n"
             "(official Stage-3 categories; seed 0 of silent sampling)", fontsize=9)
ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
ax.grid(color="#e8e7e2", lw=0.6)
for s in ax.spines.values():
    s.set_color("#bdbcb6")
ax.legend(fontsize=7, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "guideme_blind_frontier.png")
print("saved", OUT / "guideme_blind_frontier.png")
