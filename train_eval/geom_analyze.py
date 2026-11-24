import json, numpy as np, sys
R = sys.argv[1]; names = sys.argv[2:]
LAY = [7, 14, 21, 28]
def load(n):
    m = json.load(open(f"{R}/{n}.json")); z = np.load(f"{R}/{n}.npz"); return m, z["h"].astype(np.float32), z["g"]
def auc(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float(((p[:, None] > q[None]).sum() + 0.5 * (p[:, None] == q[None]).sum()) / (len(p) * len(q)))
def probe_cv(X, y, groups):
    import torch
    g = np.unique(groups); fold = {r: i % 5 for i, r in enumerate(sorted(g, key=lambda s: int(s)))}; f = np.array([fold[r] for r in groups])
    sc = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = torch.tensor((X[tr] - mu) / sd); yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3))
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt, pos_weight=pw) + 1e-2 * (w ** 2).sum(); l.backward(); return l
        opt.step(cl); sc[te] = (torch.tensor((X[te] - mu) / sd) @ w + b).detach().numpy()
    return sc
for n in names:
    m, H, G = load(n); kind = np.array([x["kind"] for x in m]); el = np.array([x["elapsed"] for x in m]); rec = np.array([x["video_id"].split("_")[0] for x in m])
    mis = kind == "mistake"; clean = kind == "clean"; suc = kind == "success"; nd = kind == "notdone"
    out = [n]
    for li, L in enumerate(LAY):
        X = H[:, li]; Xc = X - X.mean(0)
        # centroid separation (elapsed-matched pairs) in units of within-class spread
        def sep(a, b):
            d = np.linalg.norm(Xc[a].mean(0) - Xc[b].mean(0)); s = np.sqrt((Xc[a].var(0).sum() + Xc[b].var(0).sum()) / 2); return d / s
        # time direction: ridge regression of elapsed on X (fit on all), R^2
        A = np.c_[Xc, np.ones(len(X))]; w = np.linalg.solve(A.T @ A + 10 * np.eye(A.shape[1]), A.T @ el); r2 = 1 - ((A @ w - el) ** 2).sum() / ((el - el.mean()) ** 2).sum()
        sel_m = mis | clean; sel_d = suc | nd
        pm = probe_cv(X[sel_m], mis[sel_m].astype(float), rec[sel_m]); pd = probe_cv(X[sel_d], suc[sel_d].astype(float), rec[sel_d])
        out.append(f"L{L}: sep(mis|clean) {sep(mis, clean):.3f} sep(succ|notdone) {sep(suc, nd):.3f} probeAUROC mis-vs-clean {auc(pm[mis[sel_m]], pm[~mis[sel_m]]):.3f} succ-vs-notdone {auc(pd[suc[sel_d]], pd[~suc[sel_d]]):.3f} timeR2 {r2:.2f}")
    if G.size:
        P = G[:, 512:]; out.append(f"HEAD protos (val): mistake-proto AUROC mis-vs-clean {auc(P[mis, 2], P[clean, 2]):.3f} | success-proto succ-vs-notdone {auc(P[suc, 1], P[nd, 1]):.3f} | argmax acc by kind " +
                                  str({k: round(float((P[kind == k].argmax(1) == {'silence': 0, 'shifted': 0, 'notdone': 0, 'clean': 0, 'success': 1, 'mistake': 2}[k]).mean()), 2) for k in np.unique(kind)}))
    print("\n  ".join(out), flush=True)
