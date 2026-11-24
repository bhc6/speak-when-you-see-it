import json, glob, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "DATA_ROOT"; N = json.load(open(f"{D}/analysis_code/nested_frontier.json"))
C = {f.split('/')[-1][:-5]: json.load(open(f)) for f in glob.glob(f"{D}/analysis_code/opcurve/*.json")}
style = {"final__sft_prob_mc": ("Full recipe s0", "#c0392b", "-"), "seeds__final__s1_prob_mc": ("Full recipe s1", "#e74c3c", "-"),
         "seeds__final__s2_prob_mc": ("Full recipe s2", "#f1948a", "-"), "jobs__s3_prob_mc": ("Full recipe s3", "#922b21", "-"),
         "jobs__T1v4_4b_prob": ("Full recipe 4B", "#7d3c98", "-"), "jobs__ABL_noND_prob": ("− not-done", "#2e86c1", "--"),
         "jobs__ABL_noCL_prob": ("− clean", "#17a589", "--"), "jobs__ABL_none_prob": ("− both", "#566573", "--"),
         "jobs__T1v2_2400_prob": ("− both, re-weighted (T1v2)", "#b7950b", ":"), "jobs__T1p2400_prob": ("+ cross-recording pairs", "#ca6f1e", "-.")}
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, dv, suf, title in ((axs[0], "neurips2025", "_test36", "QICD test (36)"), (axs[1], "ego_mc_bench", "_mc", "Ego-MC (39, zero-shot transfer)")):
    for lvl, col, lab in (("time_text", "#e5e7e9", "blind: time + instruction priors"), ("time_only", "#ccd1d1", "blind: time only")):
        h = N[dv][lvl]; ax.fill_between([p[0] for p in h], 0, [p[1] for p in h], color=col, label=lab)
    for k, d in C.items():
        base = k.replace("_test36", "").replace("_mc", "")
        key = k if k in style else base
        if key not in style or not k.endswith(suf.replace("_test36", "_test36")) and not (suf == "_mc" and k.endswith("_mc")): continue
        if suf == "_test36" and not k.endswith("_test36"): continue
        lab, c, ls = style[key]; pts = [(x["ic"], x["f1"]) for x in d["curve"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=c, lw=1.6, label=lab); ax.scatter(*pts[0], color=c, s=40, zorder=5)
    if dv == "neurips2025":
        for lab, ic, f1, c in (("Full recipe s0 (frozen τ)", 42.0, 0.306, "#c0392b"),):
            ax.scatter(ic, f1, marker="*", s=160, color=c, zorder=6, label=lab)
    ax.set_title(title); ax.set_xlabel("IC-Acc (%)"); ax.set_ylabel("mistake F1"); ax.set_xlim(0, 72); ax.set_ylim(0, 0.42); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right")
fig.suptitle("Operating curves (exact offline sweep of the mistake threshold from the logged run; dot = logged default τ=0.35) vs. nested video-blind frontiers", fontsize=10)
plt.tight_layout(); plt.savefig(f"{D}/paper/fig/fig4_operating_curves.png", dpi=180); print("saved")
