# Elapsed-time-only AUROC per sample pool (CPU). Benchmark pools: every official 5-s check from GT step start to GT step
# end + 15 s; completion-positive = clean step and |t - end| < 15; mistake-positive = |t - f| < 15 for a GT mistake f.
import sys, os, io, contextlib, bisect
EV='WORKSPACE/qic/ref_sys/qualcomm_interactive_cooking_eval'; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
def auc(p, n):
    n = sorted(n); return sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p) / max(len(p) * len(n), 1)
for dv, sp in (("neurips2025","validation"),("neurips2025","test"),("ego_mc_bench","test")):
    with contextlib.redirect_stdout(io.StringIO()): ds = utils.load_dataset(dv, "advanced_planning", sp)
    cp, cn, mp, mn = [], [], [], []
    for a in ds.annotations:
        g = ev.get_ground_truth_data(ds, a["video_id"])
        for tx, ts, ty, hm in zip(g["texts"], g["texts_timestamps"], g["texts_types"], g["has_mistake"]):
            s0, s1 = ts[0], max(ts); mis = [t for t, y in zip(ts[1:], ty[1:]) if "mistake" in y]
            e = 5.0
            while s0 + e <= s1 + 15:
                t = s0 + e
                (cp if (not hm and abs(t - s1) < 15) else cn).append(e)
                (mp if any(abs(t - f) < 15 for f in mis) else mn).append(e)
                e += 5
    print(f"{dv} {sp}: checks {len(cp)+len(cn)} | elapsed-only AUROC completion {auc(cp,cn):.3f} (pos {len(cp)}) | mistake {auc(mp,mn):.3f} (pos {len(mp)})")
