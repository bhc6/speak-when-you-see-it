"""CPU sweep over frozen representations: which alignment objective / head / layer / data size gives a readout that
(a) detects mistakes, (b) separates mistake from completion moments, (c) does not lean on elapsed time?
Everything is recipe-grouped 5-fold CV (held-out recipes), metrics pooled over folds, video-bootstrap CI for key ones.
Datasets (frozen states already on disk, nothing re-run):
  A57   phase-A states, neutral prompt, full validation (57 videos): kinds mistake / success / silence
  G11z  geometry-probe states, official mistake prompt, zero-shot, 11 videos: + clean / notdone / shifted
  G11t  same situations, T1v3@500 backbone
  G11s  same situations, S1v3@500 backbone (if present)
Grid: layer in {7,14,21,28,cat} x head in {linear, mlp512} x objective in {bin (mistake vs rest), type_bal (3-way,
balanced weights), type_skew (3-way, SIA-like 55% silence), pair_ms (mistake vs completion only, applied to all)} x
train fraction in {0.25, 0.5, 1.0} (data-size curve for the MLP question).
Output: one JSON line per config in OUT (resumable)."""
import itertools, json, os, sys, glob
import numpy as np, torch
from multiprocessing import Pool
torch.set_num_threads(2)
R = "WORKSPACE/qic/results"; OUT = sys.argv[1]
KIND3 = {"silence": 0, "shifted": 0, "notdone": 0, "clean": 0, "success": 1, "mistake": 2}


def load():
    D = {}
    H, it = [], []
    for js in sorted(glob.glob(R + "/probe_vl_r*_shard*.json")):
        part = json.load(open(js)); h = np.load(js + ".npz")["h"]; k = 0
        for v in part:
            for x in v["items"]:
                it.append({"video_id": v["video_id"], "kind": x["type"], "elapsed": min((x["seek"] - x["buf_start"]) / 2, 120.0)}); H.append(h[k]); k += 1
    D["A57"] = (np.stack(H).astype(np.float32), it)
    for tag, f in (("G11z", "zero"), ("G11t", "T1v3"), ("G11s", "S1v3")):
        p = f"{R}/geom/{f}.npz"
        if os.path.exists(p):
            D[tag] = (np.load(p)["h"].astype(np.float32), json.load(open(f"{R}/geom/{f}.json")))
    return D


def auc(p, q):
    p, q = np.asarray(p), np.asarray(q)
    if not len(p) or not len(q): return float("nan")
    return float(((p[:, None] > q[None]).sum() + 0.5 * (p[:, None] == q[None]).sum()) / (len(p) * len(q)))


def spear(a, b):
    if len(a) < 3: return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def fit_predict(Xtr, ytr, Xte, head, obj, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd); B = torch.tensor((Xte - mu) / sd)
    nout = 1 if obj in ("bin", "pair_ms") else 3
    net = torch.nn.Linear(A.shape[1], nout) if head == "linear" else torch.nn.Sequential(torch.nn.Linear(A.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, nout))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    y = torch.tensor(ytr)
    if nout == 1:
        pw = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-3), dtype=torch.float32)
        lossf = lambda o: torch.nn.functional.binary_cross_entropy_with_logits(o[:, 0], y.float(), pos_weight=pw)
    else:
        cnt = np.bincount(ytr, minlength=3).astype(np.float32)
        w = torch.tensor(len(ytr) / (3 * np.maximum(cnt, 1)) if obj == "type_bal" else np.ones(3, np.float32))
        lossf = lambda o: torch.nn.functional.cross_entropy(o, y.long(), weight=w)
    for _ in range(300):
        opt.zero_grad(); l = lossf(net(A)); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        o = net(B)
    return (o[:, 0] if nout == 1 else (torch.log_softmax(o, -1)[:, 2] - torch.log_softmax(o, -1)[:, :2].logsumexp(-1))).numpy()


def run(cfg):
    ds, layer, head, obj, frac = cfg
    H, meta = DATA[ds]
    X = H[:, 1:].reshape(len(H), -1) if layer == "cat" else H[:, [7, 14, 21, 28].index(layer)]  # cat = layers 14, 21, 28
    kind = np.array([m["kind"] for m in meta]); el = np.array([m["elapsed"] for m in meta])
    grp = np.array([m["video_id"].split("_")[0] for m in meta]); vid = np.array([m["video_id"] for m in meta])
    y3 = np.array([KIND3[k] for k in kind]); ybin = (kind == "mistake").astype(np.int64)
    rec = sorted(set(grp), key=int); fold = {g: i % 5 for i, g in enumerate(rec)}; f = np.array([fold[g] for g in grp])
    rng = np.random.default_rng(0); score = np.full(len(kind), np.nan)
    for k in range(5):
        tr = np.where(f != k)[0]; te = np.where(f == k)[0]
        if frac < 1: tr = rng.choice(tr, max(20, int(len(tr) * frac)), replace=False)
        if obj == "pair_ms": tr = tr[np.isin(kind[tr], ["mistake", "success"])]
        ytr = ybin[tr] if obj in ("bin", "pair_ms") else y3[tr]
        if obj == "type_skew":  # resample to the SIA-v3 class mix: 55% silence-like / 20% success / 25% mistake
            idx = [np.where(ytr == c)[0] for c in range(3)]; n = len(tr)
            pick = np.concatenate([rng.choice(idx[c], int(n * p), replace=True) for c, p in enumerate([.55, .20, .25]) if len(idx[c])])
            tr, ytr = tr[pick], ytr[pick]
        score[te] = fit_predict(X[tr], ytr, X[te], head, obj, k)
    neg = kind != "mistake"
    res = {"ds": ds, "layer": str(layer), "head": head, "obj": obj, "frac": frac, "n": int(len(kind)),
           "mis_all": auc(score[kind == "mistake"], score[neg]), "mis_vs_success": auc(score[kind == "mistake"], score[kind == "success"]),
           "mis_vs_silence": auc(score[kind == "mistake"], score[kind == "silence"]),
           "rho_time_neg": spear(score[neg], el[neg])}
    if "clean" in kind: res["mis_vs_clean"] = auc(score[kind == "mistake"], score[kind == "clean"])
    # video bootstrap CI for mis_all
    uv = np.unique(vid); b = []
    for _ in range(300):
        s = set(rng.choice(uv, len(uv)).tolist()); m = np.isin(vid, list(s))
        b.append(auc(score[m & (kind == "mistake")], score[m & neg]))
    res["mis_all_ci"] = [round(float(x), 3) for x in np.nanpercentile(b, [2.5, 97.5])]
    return res


DATA = load()
if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): r = json.loads(l); done.add((r["ds"], r["layer"], r["head"], r["obj"], r["frac"]))
    cfgs = [c for c in itertools.product(list(DATA), [14, 21, 28, "cat"], ["linear", "mlp512"], ["bin", "type_bal", "type_skew", "pair_ms"], [1.0, 0.5, 0.25])
            if (c[0], str(c[1]), c[2], c[3], c[4]) not in done and not (c[4] < 1 and c[1] not in (21, "cat"))]
    print("datasets", {k: len(v[1]) for k, v in DATA.items()}, "configs", len(cfgs), flush=True)
    with Pool(int(os.environ.get("NW", "12"))) as p, open(OUT, "a") as fo:
        for r in p.imap_unordered(run, cfgs):
            fo.write(json.dumps(r) + "\n"); fo.flush()
    print("sweep done", flush=True)
