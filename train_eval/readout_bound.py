"""CPU only. Can a frozen zero-shot layer + linear probe beat the models' own outputs on the same situations?
Situations: validation subset (11 videos), GT-sampled (geom probe set). Probe: logistic regression, recipe-grouped
5-fold CV. Output scores: first-token P(yes) from the AUROC runs (zero-shot; T1v3/S1v3 at step 300), matched by
(video, step, kind, elapsed). Tasks use the output runs' negative set (silence/success/shifted or silence/mistake/shifted)."""
import json, sys
import numpy as np, torch
G = "WORKSPACE/qic/results/train3/geom/"; A = "WORKSPACE/qic/results/train3/auroc/"
def auc(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float(((p[:, None] > q[None]).sum() + 0.5 * (p[:, None] == q[None]).sum()) / (len(p) * len(q)))
def prec_at_recall(s, y, r):
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); k = np.searchsorted(tp, r * ys.sum()) + 1
    return float(tp[k - 1] / k)
def cv(X, y, groups, resid=None):
    rec = sorted(set(groups), key=int); fold = {g: i % 5 for i, g in enumerate(rec)}; f = np.array([fold[g] for g in groups]); sc = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k; Xtr, Xte = X[tr], X[te]
        if resid is not None:  # remove what elapsed time explains (fit on train fold)
            Z = np.c_[resid, resid ** 2, np.ones(len(resid))]; W = np.linalg.lstsq(Z[tr], Xtr, rcond=None)[0]; Xtr, Xte = Xtr - Z[tr] @ W, Xte - Z[te] @ W
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xa = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True); pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3))
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w + b, yt, pos_weight=pw) + 1e-2 * (w ** 2).sum(); l.backward(); return l
        opt.step(cl); sc[te] = (torch.tensor((Xte - mu) / sd, dtype=torch.float32) @ w + b).detach().numpy()
    return sc
meta = json.load(open(G + "zero.json")); H = np.load(G + "zero.npz")["h"].astype(np.float32)
key = lambda x: (x["video_id"], x["step"], x["kind"], round(x["elapsed"], 2))
outs = {n: {key(x): x for x in json.load(open(A + n + ".json"))} for n in ("zero", "T1v3_s300", "S1v3_s300", "T1v2_s300")}
idx = [i for i, x in enumerate(meta) if key(x) in outs["zero"]]
m = [meta[i] for i in idx]; Hs = H[idx]; kind = np.array([x["kind"] for x in m]); el = np.array([x["elapsed"] for x in m]); grp = np.array([x["video_id"].split("_")[0] for x in m])
print(f"matched situations {len(m)} / geom {len(meta)}; kinds", {k: int((kind == k).sum()) for k in np.unique(kind)})
for task, pos, field in (("MISTAKE", "mistake", "p_mis"), ("DONE", "success", "p_done")):
    y = (kind == pos).astype(float)
    print(f"\n== {task} (pos {int(y.sum())} / neg {int((1 - y).sum())})")
    print(f"  elapsed-only AUROC {auc(el[y == 1], el[y == 0]):.3f}")
    for n, d in outs.items():
        s = np.array([d[key(x)][field] for x in m])
        print(f"  output {n:10s} AUROC {auc(s[y == 1], s[y == 0]):.3f}  P@R0.3 {prec_at_recall(s, y, .3):.3f}  P@R0.5 {prec_at_recall(s, y, .5):.3f}")
    for li, L in enumerate([7, 14, 21, 28]):
        s = cv(Hs[:, li], y, grp); s2 = cv(Hs[:, li], y, grp, resid=el)
        print(f"  zero-shot L{L} probe AUROC {auc(s[y == 1], s[y == 0]):.3f}  P@R0.3 {prec_at_recall(s, y, .3):.3f}  P@R0.5 {prec_at_recall(s, y, .5):.3f} | time-residualised {auc(s2[y == 1], s2[y == 0]):.3f}")

print("\n== MISTAKE vs each negative kind (probe trained on all; zero-shot layers) / outputs")
y = (kind == "mistake").astype(float)
for li, L in enumerate([14, 21, 28]):
    s = cv(Hs[:, [7, 14, 21, 28].index(L)], y, grp)
    print(f"  L{L} probe:", {k: round(auc(s[kind == 'mistake'], s[kind == k]), 3) for k in ('silence', 'success', 'shifted')})
for n, d in outs.items():
    s = np.array([d[key(x)]["p_mis"] for x in m]); print(f"  output {n}:", {k: round(auc(s[kind == 'mistake'], s[kind == k]), 3) for k in ('silence', 'success', 'shifted')})
# probe trained WITHOUT completion moments as negatives, tested on all (does including success in training help?)
