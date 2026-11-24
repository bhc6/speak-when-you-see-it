"""Controls for the phase-A linear probe (is it just reading clip length / timing?).
python probe_controls.py RESULTS_DIR LAYER_INDEX
C0 timing-only probe: features = window length, time since step start, log of both (no video, no text)
C1 length-matched probe: subsample so that the three classes have the same window-length distribution (2 s bins,
   min count per bin), then the same grouped-CV linear probe on hidden states vs on timing features
C2 residual probe: hidden states with window-length features linearly regressed out (fit on train folds)
C3 selectivity control (Hewitt & Liang 2019): same probe on labels permuted within each length bin
All: recipe-grouped 5-fold CV, balanced accuracy, video bootstrap 95% CI."""
import glob, json, sys
import numpy as np, torch

R, li = sys.argv[1], int(sys.argv[2])
H, items = [], []
for js in sorted(glob.glob(f"{R}/probe_vl_r*_shard*.json")):
    part = json.load(open(js)); h = np.load(js + ".npz")["h"]; k = 0
    for v in part:
        for it in v["items"]:
            items.append(dict(it, video_id=v["video_id"])); H.append(h[k, li]); k += 1
X = np.stack(H).astype(np.float32)
lab = np.array([{"silence": 0, "success": 1, "mistake": 2}[it["type"]] for it in items])
L = np.array([(it["seek"] - it["buf_start"]) / 2 for it in items])                     # seconds in the window
E = np.array([max(it["seek"] / 2 - (it["buf_start"] / 2), 0) for it in items])          # == L unless capped at 120 s
T = np.array([it["t"] - (it["buf_start"] / 2) for it in items])
F = np.stack([L, np.log1p(L), np.minimum(L, 120) == 120, T], 1).astype(np.float32)
video = np.array([it["video_id"] for it in items]); recipe = np.array([v.split("_")[0] for v in video])
fold_of = {r: i % 5 for i, r in enumerate(sorted(set(recipe), key=int))}; fold = np.array([fold_of[r] for r in recipe])
uv = np.unique(video); rows = {v: np.where(video == v)[0] for v in uv}; rng = np.random.default_rng(0)
B = [rng.choice(len(uv), len(uv)) for _ in range(1000)]


def bal(pred, y, idx):
    return np.mean([np.mean(pred[idx][y[idx] == c] == c) for c in range(3) if np.any(y[idx] == c)])


def ci(pred, y, sel):
    vals = []
    for b in B:
        idx = np.concatenate([rows[uv[j]] for j in b]); idx = idx[sel[idx]]
        if len(idx): vals.append(bal(pred, y, idx))
    return np.percentile(vals, [2.5, 97.5])


def probe(Z, y, sel, resid=None):
    pred = np.full(len(y), -1)
    for k in range(5):
        tr, te = sel & (fold != k), sel & (fold == k)
        A, Bt = Z[tr], Z[te]
        if resid is not None:  # regress window-length features out of Z, fit on train
            G = np.c_[resid, np.ones(len(resid))]; W = np.linalg.lstsq(G[tr], A, rcond=None)[0]
            A, Bt = (A - G[tr] @ W).astype(np.float32), (Bt - G[te] @ W).astype(np.float32)
        A, Bt = A.astype(np.float32), Bt.astype(np.float32); mu, sd = A.mean(0), A.std(0) + 1e-6
        At = torch.tensor((A - mu) / sd); yt = torch.tensor(y[tr])
        w = torch.zeros(Z.shape[1], 3, requires_grad=True); b = torch.zeros(3, requires_grad=True)
        cw = torch.tensor(len(yt) / (3 * np.bincount(y[tr], minlength=3) + 1e-6), dtype=torch.float32)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(At @ w + b, yt, weight=cw) + 1e-2 * (w ** 2).sum(); l.backward(); return l
        opt.step(cl)
        pred[te] = (torch.tensor((Bt - mu) / sd) @ w + b).argmax(1).numpy()
    s = bal(pred, y, np.where(sel)[0]); lo, hi = ci(pred, y, sel)
    return f"{s:.3f} [{lo:.3f},{hi:.3f}]"


allm = np.ones(len(lab), bool)
print(f"layer index {li}; n={len(lab)}; window length median by class (s):",
      {c: float(np.median(L[lab == i])) for i, c in enumerate(["silence", "success", "mistake"])})
print("C0 timing-only probe           ", probe(F, lab, allm))
print("   hidden-state probe (all)    ", probe(X, lab, allm))
# C1 length-matched subsample
bins = np.floor(np.minimum(L, 120) / 2).astype(int); keep = np.zeros(len(lab), bool)
for bb in np.unique(bins):
    idx = [np.where((bins == bb) & (lab == c))[0] for c in range(3)]
    m = min(len(i) for i in idx)
    for i in idx:
        if m: keep[rng.choice(i, m, replace=False)] = True
print(f"C1 length-matched (n={keep.sum()}, per class {np.bincount(lab[keep])})")
print("   timing-only                 ", probe(F, lab, keep))
print("   hidden-state                ", probe(X, lab, keep))
print("C2 hidden-state, length regressed out", probe(X, lab, allm, resid=F))
perm = lab.copy()
for bb in np.unique(bins):
    i = np.where(bins == bb)[0]; perm[i] = rng.permutation(lab[i])
print("C3 selectivity: labels permuted within length bins", probe(X, perm, allm))
