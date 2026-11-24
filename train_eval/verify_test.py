import json, sys, io, re, random, contextlib, collections
EV = sys.argv[1]; sys.path.insert(0, EV); import os; os.chdir(EV)
import eval as ev, utils
ds = utils.load_dataset(sys.argv[3], "advanced_planning", "test")
for path in sys.argv[4:]:
    P = json.load(open(path))
    vids = [p["video_id"] for p in P]
    # sanity
    texts = [t for p in P for t in p["pred_texts"]]
    kinds = collections.Counter(t.split(":")[0] for t in texts)
    fbs = [t for t in texts if t.startswith("Feedback")]
    dup = sum(c - 1 for c in collections.Counter(fbs).values() if c > 1)
    mono = all(all(a <= b + 1e-6 for a, b in zip(p["pred_timestamps"], p["pred_timestamps"][1:])) for p in P)
    # official eval, per-video cumulative prints
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(ds, P)
    o = buf.getvalue()
    C = [tuple(map(int, m)) for m in re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)]
    ICs = [float(x) for x in re.findall(r"IC-Acc: ([\d.]+)", o)]
    # per-video increments; IC needs numerator/denominator: denominator per video = clean steps
    clean = []
    for v in vids:
        g = ev.get_ground_truth_data(ds, v); clean.append(sum(not x for x in g["has_mistake"]))
    cumc = [sum(clean[:i+1]) for i in range(len(clean))]
    det = [round(ic * c / 100) for ic, c in zip(ICs, cumc)]
    per = []
    prev = (0, 0, 0, 0); pd = 0
    for i in range(len(vids)):
        d = tuple(a - b for a, b in zip(C[i], prev)); per.append((d, det[i] - pd, clean[i])); prev = C[i]; pd = det[i]
    def met(S):
        tp, fp, tn, fn = (sum(x[0][k] for x in S) for k in range(4)); dt = sum(x[1] for x in S); cl = sum(x[2] for x in S)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return 100 * dt / max(cl, 1), p, r, 2 * p * r / max(p + r, 1e-9)
    full = met(per); rng = random.Random(0); B = sorted((met([rng.choice(per) for _ in per]) for _ in range(2000)), key=lambda x: x[3])
    Bi = sorted(b[0] for b in B)
    print(f"{path.split('/')[-2]}: videos {len(set(vids))}/{len(ds)} kinds {dict(kinds)} dup_fb {dup} monotone_ts {mono}")
    print(f"  official final: IC {ICs[-1]} counts {C[-1]} | recomputed IC {full[0]:.1f} P {full[1]:.3f} R {full[2]:.3f} F1 {full[3]:.3f}")
    print(f"  95% video bootstrap: F1 [{B[50][3]:.3f}, {B[1949][3]:.3f}]  IC [{Bi[50]:.1f}, {Bi[1949]:.1f}]")
