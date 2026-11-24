"""Paired per-video bootstrap of official-eval metrics between two prediction files on the same split (official eval,
per-video counts from its cumulative prints). python paired_test.py EVAL_DIR DATASET_VERSION SPLIT A.json B.json"""
import contextlib, io, json, os, random, re, sys
EV, DV, SP, FA, FB = sys.argv[1:6]; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
ds = utils.load_dataset(DV, "advanced_planning", SP)
def per_video(P):
    """one official run over all videos; per-video counts = differences of its cumulative prints (videos ordered so the
    first one has a mistake-free step: the eval divides by the running clean-step count)."""
    cl = {p["video_id"]: sum(not x for x in ev.get_ground_truth_data(ds, p["video_id"])["has_mistake"]) for p in P}
    P = sorted(P, key=lambda p: (cl[p["video_id"]] == 0, p["video_id"]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(ds, P)
    o = buf.getvalue()
    C = [tuple(map(int, m)) for m in re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)]
    I = [float(x) for x in re.findall(r"IC-Acc: ([\d.]+)", o)]
    out, prev, pdet, cum = {}, (0, 0, 0, 0), 0, 0
    for p, c, ic in zip(P, C, I):
        v = p["video_id"]; cum += cl[v]; det = round(ic * cum / 100)
        out[v] = (tuple(x - y for x, y in zip(c, prev)), det - pdet, cl[v]); prev, pdet = c, det
    assert len(out) == len(P)
    return out
A, B = per_video(json.load(open(FA))), per_video(json.load(open(FB)))
vids = sorted(set(A) & set(B))
def met(D, V):
    tp, fp, tn, fn = (sum(D[v][0][k] for v in V) for k in range(4)); det = sum(D[v][1] for v in V); cl = sum(D[v][2] for v in V)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return 100 * det / max(cl, 1), 2 * p * r / max(p + r, 1e-9)
a, b = met(A, vids), met(B, vids); rng = random.Random(0); dI, dF = [], []
for _ in range(5000):
    V = [rng.choice(vids) for _ in vids]; x, y = met(A, V), met(B, V); dI.append(y[0] - x[0]); dF.append(y[1] - x[1])
dI.sort(); dF.sort(); q = lambda L, p: L[int(p * (len(L) - 1))]
MI, MF = float(os.environ.get("EQ_IC", "5")), float(os.environ.get("EQ_F1", "0.05"))
eqI = -MI < q(dI, .05) and q(dI, .95) < MI; eqF = -MF < q(dF, .05) and q(dF, .95) < MF
pI = 2 * min(sum(x <= 0 for x in dI), sum(x >= 0 for x in dI)) / len(dI); pF = 2 * min(sum(x <= 0 for x in dF), sum(x >= 0 for x in dF)) / len(dF)
print(f"videos {len(vids)} | A {a[0]:.1f}/{a[1]:.3f} B {b[0]:.1f}/{b[1]:.3f} | dIC {b[0]-a[0]:+.1f} 95%[{q(dI,.025):+.1f},{q(dI,.975):+.1f}] 90%[{q(dI,.05):+.1f},{q(dI,.95):+.1f}] p={min(pI,1):.4f} TOST(+-{MI}) {'EQUIV' if eqI else 'not-equiv'} | dF1 {b[1]-a[1]:+.3f} 95%[{q(dF,.025):+.3f},{q(dF,.975):+.3f}] 90%[{q(dF,.05):+.3f},{q(dF,.95):+.3f}] p={min(pF,1):.4f} TOST(+-{MF}) {'EQUIV' if eqF else 'not-equiv'}")
