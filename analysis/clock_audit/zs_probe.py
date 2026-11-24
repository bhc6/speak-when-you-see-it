"""Why the zero-shot model raises alarms: mean probability of "yes" to the mistake question on the fixed validation situations
of the grounding probe (gprobe outputs, 1,233 situations), split by situation kind, under the original window, black frames
(evidence removed, frame count kept) and pad60 (120 copies of the first frame prepended: elapsed-time cue +60 s, content kept).
Compares zero-shot Qwen3-VL-2B (single check, official prompt) with fine-tuned variants. Also the AUROC of the mistake
question (mistake vs clean or silent situations), as in the supplement's probe table.
python3 zs_probe.py -> out/zs_probe.json"""
import json, glob, os, statistics as st, bisect
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
best = {}
for f in glob.glob(V0 + "*/results/jobs/gprobe/*.json"):
    k = os.path.basename(f)[:-5]
    try: d = json.load(open(f))
    except Exception: continue
    if k not in best or len(d) > len(best[k]): best[k] = d
NAMES = {"R0_zs": "Qwen3-VL-2B zero-shot", "ABL_none": "no paired neg. s0", "T1v4_s0": "ECT s0"}
NEG = ("silence", "clean")


def auc(p, n):
    n = sorted(n); return sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p) / max(len(p) * len(n), 1)


res = {}
for k, name in NAMES.items():
    d = best[k]
    m = lambda c, kinds: round(st.mean(x[f"mis_{c}"] for x in d if x["kind"] in kinds and x.get(f"mis_{c}") is not None), 3)
    res[name] = dict(n=len(d), mis_yes_no_mistake=m("orig", NEG), mis_yes_mistake=m("orig", ("mistake",)),
                     mis_yes_no_mistake_black=m("black", NEG), mis_yes_no_mistake_pad60=m("pad60", NEG),
                     auroc_mistake=round(auc([x["mis_orig"] for x in d if x["kind"] == "mistake"], [x["mis_orig"] for x in d if x["kind"] in NEG]), 3))
    full = [x for x in d if x["kind"] in NEG and x["n"] <= 120]   # windows that receive the full 60 s of padding (cut to 240 frames)
    res[name].update(n_no_mistake_full_pad=len(full), mis_yes_no_mistake_full_pad_orig=round(st.mean(x["mis_orig"] for x in full), 3),
                     mis_yes_no_mistake_full_pad_pad60=round(st.mean(x["mis_pad60"] for x in full), 3))
    print(name, res[name])
json.dump(res, open(HERE + "/out/zs_probe.json", "w"), indent=1)
