"""One line per finished AUROC file in results/train3/auroc: mistake/done AUROC and mean P(yes) pos/neg."""
import glob, json, os
import numpy as np


def auc(p, n):
    p, n = np.array(p), np.array(n)
    if not len(p) or not len(n): return float("nan")
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


R = "WORKSPACE/qic/results/train3/auroc"
for f in sorted(glob.glob(R + "/*.json")):
    n = os.path.basename(f)[:-5]
    if not os.path.exists(f"{R}/../.stage_auroc_{n}.done"): continue
    d = json.load(open(f))
    pm = [x["p_mis"] for x in d if x["kind"] == "mistake"]; nm = [x["p_mis"] for x in d if x["kind"] != "mistake"]
    pd = [x["p_done"] for x in d if x["kind"] == "success"]; nd = [x["p_done"] for x in d if x["kind"] != "success"]
    print(f"AUROC {n}: mistake {auc(pm, nm):.3f} (P {np.mean(pm):.2f}/{np.mean(nm):.2f}) done {auc(pd, nd):.3f} (P {np.mean(pd):.2f}/{np.mean(nd):.2f}) n={len(d)}")
