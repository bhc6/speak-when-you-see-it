"""Phase A analysis (CPU): does a frozen VLM's situation state already point at the right intervention?
python probe_analyze.py RESULTS_DIR   (expects probe_vl.json + probe_vl_r*_shard*.json.npz, probe/text_*.npz|json)
Reports per layer, with video-level bootstrap 95% CIs:
  Z1 zero-shot 3-way type (silence / success / mistake) by cosine to the silence text, the success text and the
     nearest mistake candidate; balanced accuracy (chance 0.333)
  Z2 zero-shot mistake retrieval: rank of the GT feedback text among all 813 mistake candidates and among
     candidates of the same activity; R@1/R@10/MRR, chance given
  L1 linear probe, recipe-grouped 5-fold CV, 3-way type, balanced accuracy: video situation vs text-only situation
Cosine is computed after removing each modality's mean (anisotropy); raw cosine also reported for Z1."""
import glob, json, os, sys
import numpy as np, torch

R = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "probe_vl"; P = f"{R}/" + (sys.argv[3] if len(sys.argv) > 3 else "probe")
LAYERS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [7, 14, 21, 28]
# rows of each shard npz follow that shard's own json order
H, items = [], []
for js in sorted(glob.glob(f"{R}/{TAG}_r*_shard*.json")):
    part = json.load(open(js)); h = np.load(js + ".npz")["h"]; k = 0
    for v in part:
        for it in v["items"]:
            it = dict(it, video_id=v["video_id"]); items.append(it); H.append(h[k]); k += 1
    assert k == len(h), js
H = np.stack(H).astype(np.float32)
C = json.load(open(f"{P}/text_candidates.json")); CH = np.load(f"{P}/text_candidates.npz")["h"].astype(np.float32)
TS = json.load(open(f"{P}/text_situations.json")); TH = np.load(f"{P}/text_situations.npz")["h"].astype(np.float32)
key = lambda it: (it["video_id"], it["step"], it["type"], round(it["t"], 2))
tidx = {key(it): i for i, it in enumerate(TS)}
TH = TH[[tidx[key(it)] for it in items]]
types = np.array([it["type"] for it in items]); lab = np.array([{"silence": 0, "success": 1, "mistake": 2}[t] for t in types])
video = np.array([it["video_id"] for it in items]); recipe = np.array([v.split("_")[0] for v in video])
ck = np.array([c["kind"] for c in C]); mis_idx = np.where(ck == "mistake")[0]
i_succ, i_sil = int(np.where(ck == "success")[0][0]), int(np.where(ck == "silence")[0][0])
text2c = {C[i]["text"]: i for i in mis_idx}; act = np.array([C[i].get("activity", "") for i in range(len(C))])
v2act = {c["video_id"]: c["activity"] for c in C if "video_id" in c}
print("situations", len(items), dict(zip(*np.unique(types, return_counts=True))), "candidates", len(C))
rng = np.random.default_rng(0); uv = np.unique(video)
bidx = [rng.choice(len(uv), len(uv)) for _ in range(2000)]
rows_of = {v: np.where(video == v)[0] for v in uv}


def boot(per_item, agg):
    vals = [agg(np.concatenate([per_item_idx for per_item_idx in (rows_of[uv[j]] for j in b)])) for b in bidx]
    return np.percentile(vals, [2.5, 97.5])


def bal_acc(pred, idx=None):
    idx = np.arange(len(lab)) if idx is None else idx
    return np.mean([np.mean(pred[idx][lab[idx] == c] == c) for c in range(3) if np.any(lab[idx] == c)])


def norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)


for li, L in enumerate(LAYERS):
    S, Cc = H[:, li], CH[:, li]
    for centered in (False, True):
        s = norm(S - S.mean(0)) if centered else norm(S); c = norm(Cc - Cc.mean(0)) if centered else norm(Cc)
        sim = s @ c.T
        pred = np.stack([sim[:, i_sil], sim[:, i_succ], sim[:, mis_idx].max(1)], 1).argmax(1)
        ba = bal_acc(pred); lo, hi = boot(None, lambda ix: bal_acc(pred, ix))
        print(f"L{L} {'centered' if centered else 'raw     '} Z1 3-way bal.acc {ba:.3f} [{lo:.3f},{hi:.3f}] (chance 0.333)  pred dist {np.bincount(pred, minlength=3)}")
    # Z2 retrieval (centered)
    s = norm(S - S.mean(0)); c = norm(Cc - Cc.mean(0)); sim = s @ c.T
    ranks, ranks_act, n_act = [], [], []
    for i in np.where(lab == 2)[0]:
        g = text2c.get(items[i]["text"])
        if g is None: continue
        sc = sim[i, mis_idx]; r = 1 + np.sum(sc > sim[i, g]); ranks.append(r)
        same = mis_idx[act[mis_idx] == v2act.get(items[i]["video_id"], "")]
        if g in same:
            sc2 = sim[i, same]; ranks_act.append(1 + np.sum(sc2 > sim[i, g])); n_act.append(len(same))
    ranks, ranks_act = np.array(ranks), np.array(ranks_act)
    print(f"L{L} Z2 all-bank (n={len(ranks)}, bank {len(mis_idx)}): R@1 {np.mean(ranks==1):.3f} R@10 {np.mean(ranks<=10):.3f} MRR {np.mean(1/ranks):.3f} median rank {np.median(ranks):.0f} (chance median ~{len(mis_idx)//2})")
    print(f"L{L} Z2 same-activity (n={len(ranks_act)}, mean bank {np.mean(n_act):.0f}): R@1 {np.mean(ranks_act==1):.3f} MRR {np.mean(1/ranks_act):.3f} chance R@1 {np.mean(1/np.array(n_act)):.3f}")
    # L1 linear probe, grouped CV by recipe
    for name, X in (("video", S), ("text-only", TH[:, li])):
        rec = np.unique(recipe); fold = {r: i % 5 for i, r in enumerate(sorted(rec, key=int))}
        f = np.array([fold[r] for r in recipe]); pred = np.zeros(len(lab), int)
        for k in range(5):
            tr, te = f != k, f == k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xt = torch.tensor((X[tr] - mu) / sd); yt = torch.tensor(lab[tr])
            w = torch.zeros(X.shape[1], 3, requires_grad=True); b = torch.zeros(3, requires_grad=True)
            cw = torch.tensor(len(yt) / (3 * np.bincount(lab[tr], minlength=3) + 1e-6), dtype=torch.float32)
            opt = torch.optim.LBFGS([w, b], max_iter=200)
            def closure():
                opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xt @ w + b, yt, weight=cw) + 1e-2 * (w ** 2).sum(); loss.backward(); return loss
            opt.step(closure)
            pred[te] = (torch.tensor((X[te] - mu) / sd) @ w + b).argmax(1).numpy()
        ba = bal_acc(pred); lo, hi = boot(None, lambda ix: bal_acc(pred, ix))
        print(f"L{L} L1 linear probe [{name:9s}] 3-way bal.acc {ba:.3f} [{lo:.3f},{hi:.3f}]")
