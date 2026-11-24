"""Does the frozen representation 'know' what the model fails to say?
Join phase-A probe situations with the model's own decisions from dense_vl_base (same windows: GT step start ->
check time; nearest dense check within 2.5 s). Model decision per check, using the baseline order:
completion YES -> success; else mistake YES -> mistake; else silence (text rule; 'p' rule also reported).
Reports: model-vs-GT agreement; GT-label probe vs model-label probe (recipe-grouped 5-fold CV); and, on
situations where the model's own decision is wrong, how often the GT-trained probe is right ('knows but doesn't
say'). python probe_vs_behavior.py RESULTS_DIR LAYER_INDEX"""
import glob, json, sys
import numpy as np, torch

R, li = sys.argv[1], int(sys.argv[2])
H, items = [], []
for js in sorted(glob.glob(f"{R}/probe_vl_r*_shard*.json")):
    part = json.load(open(js)); h = np.load(js + ".npz")["h"]; k = 0
    for v in part:
        for it in v["items"]:
            items.append(dict(it, video_id=v["video_id"])); H.append(h[k, li]); k += 1
dense = {v["video_id"]: v for v in json.load(open(f"{R}/analysis/dense_vl_base.json"))}
keep, X, gt, mod_t, mod_p = [], [], [], [], []
C = {"silence": 0, "success": 1, "mistake": 2}
for it, h in zip(items, H):
    st = dense[it["video_id"]]["steps"][it["step"]]
    if not st["checks"]: continue
    c = min(st["checks"], key=lambda c: abs(c["seek"] - it["seek"]))
    if abs(c["seek"] - it["seek"]) > 5: continue
    t_lab = 1 if "yes" in c["done_text"].lower() else (2 if "yes" in c["mis_text"].lower() else 0)
    pd = c["p_done"][0] / max(sum(c["p_done"]), 1e-9); pm = c["p_mis"][0] / max(sum(c["p_mis"]), 1e-9)
    p_lab = 1 if pd >= .5 else (2 if pm >= .5 else 0)
    X.append(h); gt.append(C[it["type"]]); mod_t.append(t_lab); mod_p.append(p_lab); keep.append(it)
X = np.stack(X).astype(np.float32); gt, mod_t, mod_p = map(np.array, (gt, mod_t, mod_p))
recipe = np.array([it["video_id"].split("_")[0] for it in keep])
fo = {r: i % 5 for i, r in enumerate(sorted(set(recipe), key=int))}; fold = np.array([fo[r] for r in recipe])
bal = lambda p, y, m=None: np.mean([np.mean(p[(y == c) & (m if m is not None else True)] == c) for c in range(3) if np.any((y == c) & (m if m is not None else True))])


def cvprobe(y):
    pred = np.zeros(len(y), int)
    for k in range(5):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; At = torch.tensor((X[tr] - mu) / sd); yt = torch.tensor(y[tr])
        w = torch.zeros(X.shape[1], 3, requires_grad=True); b = torch.zeros(3, requires_grad=True)
        cw = torch.tensor(len(yt) / (3 * np.bincount(y[tr], minlength=3) + 1e-6), dtype=torch.float32)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(At @ w + b, yt, weight=cw) + 1e-2 * (w ** 2).sum(); l.backward(); return l
        opt.step(cl); pred[te] = (torch.tensor((X[te] - mu) / sd) @ w + b).argmax(1).numpy()
    return pred


print(f"layer index {li}: matched situations {len(gt)} / {len(items)}; GT class counts {np.bincount(gt)}")
for name, m in (("text rule", mod_t), ("p>=0.5 rule", mod_p)):
    cm = np.zeros((3, 3), int)
    for a, b in zip(gt, m): cm[a, b] += 1
    print(f"model decision ({name}) vs GT: balanced acc {bal(m, gt):.3f}; confusion rows=GT[sil,succ,mis] cols=model\n{cm}")
pg = cvprobe(gt); pm = cvprobe(mod_t)
print(f"probe trained on GT labels   -> bal.acc vs GT {bal(pg, gt):.3f}")
print(f"probe trained on model labels-> bal.acc vs model output {bal(pm, mod_t):.3f}, vs GT {bal(pm, gt):.3f}")
wrong = mod_t != gt
print(f"situations where the model is wrong: {wrong.sum()} ({wrong.mean():.0%}); GT-probe correct on them: {np.mean(pg[wrong] == gt[wrong]):.3f}"
      f"; per GT class: " + ", ".join(f"{n} {np.mean(pg[wrong & (gt == c)] == c):.2f} (n={np.sum(wrong & (gt == c))})" for c, n in enumerate(["silence", "success", "mistake"])))
