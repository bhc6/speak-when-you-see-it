"""Grounding plane summary from grounding_probe.py outputs (validation, fixed situations, 5 conditions).
vis = mean|p_orig - p_swap|, black = mean|p_orig - p_black|, time = mean|p_orig - p_pad60|, for the completion (done) and
mistake (mis) prompts; AUROC of p_orig: done on success-vs-notdone situations, mis on mistake-vs-clean situations."""
import json, glob, os, collections
def auroc(pos, neg):
    if not pos or not neg: return None
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg]); r = 0; rank = 0; i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]: j += 1
        avg = (i + j + 1) / 2; r += avg * sum(1 for k in range(i, j) if s[k][1]); i = j
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
best = {}
for f in glob.glob("DATA_ROOT/*/results/jobs/gprobe/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    k = os.path.basename(f)[:-5]
    if k not in best or len(d) > len(best[k]): best[k] = d
rows = []
for k, d in sorted(best.items()):
    m = lambda a, b: sum(abs(x[a] - x[b]) for x in d if x.get(a) is not None and x.get(b) is not None) / max(1, sum(1 for x in d if x.get(a) is not None and x.get(b) is not None))
    kinds = collections.Counter(x["kind"] for x in d)
    ad = auroc([x["done_orig"] for x in d if x["kind"] == "success"], [x["done_orig"] for x in d if x["kind"] in ("notdone", "silence")])
    am = auroc([x["mis_orig"] for x in d if x["kind"] == "mistake"], [x["mis_orig"] for x in d if x["kind"] in ("clean", "silence")])
    rows.append(dict(model=k, n=len(d), done_vis=m("done_orig", "done_swap"), done_black=m("done_orig", "done_black"), done_time=m("done_orig", "done_pad60"),
                     mis_vis=m("mis_orig", "mis_swap"), mis_black=m("mis_orig", "mis_black"), mis_time=m("mis_orig", "mis_pad60"), auroc_done=ad, auroc_mis=am, kinds=dict(kinds)))
json.dump(rows, open("DATA_ROOT/audit_1114/gprobe_summary.json", "w"), indent=1)
print(f"{'model':20s} {'n':>5} done:vis black time AUROC | mis:vis black time AUROC")
for r in rows:
    f = lambda v: "  -  " if v is None else f"{v:.3f}"
    print(f"{r['model']:20s} {r['n']:5d} {f(r['done_vis'])} {f(r['done_black'])} {f(r['done_time'])} {f(r['auroc_done'])} | {f(r['mis_vis'])} {f(r['mis_black'])} {f(r['mis_time'])} {f(r['auroc_mis'])}")
