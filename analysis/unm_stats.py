"""Coverage audit 11-17: how well elapsed time (and remaining time) separates positives from their negatives, for time-matched
vs random-moment negatives (sia_data.build_v3 unmatched=False/True), seeds 0 and 1. CPU only, reads annotations."""
import sys, json; sys.path.insert(0, "WORKSPACE/qic/experiments/plan_repr")
import sia_data as D, collections, statistics as st
FR = "WORKSPACE/qicFRAMES_ROOT"
def auc(p, n):
    a = sorted([(x, 1) for x in p] + [(x, 0) for x in n]); s = 0; i = 0
    while i < len(a):
        j = i
        while j < len(a) and a[j][0] == a[i][0]: j += 1
        s += sum((i + j + 1) / 2 for k in range(i, j) if a[k][1]); i = j
    return (s - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
out = {}
for seed in (0, 1):
    for um in (False, True):
        ex, _ = D.build_v3(FR, "train", seed=seed, comist=False, unmatched=um)
        K = collections.defaultdict(list)
        for e in ex: K[e["kind"]].append(e)
        r = {"n": {k: len(v) for k, v in K.items()}}
        for pos, neg in (("success", "notdone"), ("mistake", "clean")):
            el = lambda L: [e["t"] - e["start"] for e in L]
            r[f"{pos}_vs_{neg}"] = dict(elapsed_auroc=round(auc(el(K[pos]), el(K[neg])), 3),
                                        neg_elapsed_median=round(st.median(el(K[neg])), 1), pos_elapsed_median=round(st.median(el(K[pos])), 1),
                                        neg_remaining_median=round(st.median([e["end"] - e["t"] for e in K[neg]]), 1))
        out[f"seed{seed}_{'random' if um else 'matched'}"] = r
json.dump(out, open("TMP_PATH", "w"), indent=1); print(json.dumps(out, indent=1))
