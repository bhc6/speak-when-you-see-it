"""Verify the step-advance rule of the official turn-based protocol on real runs: in every recording the number of instructions
issued equals min(confirmations + 1, number of GT steps), i.e. the next instruction comes only after a confirmation. Also counts,
per run, recordings that stop before the last step and steps reached. python check_advance.py -> out/advance.json"""
import sys, os, json, types
if "torch" not in sys.modules:  # official data.py only subclasses torch.utils.data.Dataset
    _t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
    sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
sys.path.insert(0, "DATA_ROOT/analysis_code/blind_strong"); import local_eval as L
B = "DATA_ROOT/"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
RUNS = {"zero-shot": B + "results/test_vl_mem.json", "no paired neg.": B + "results/jobs/ABL_none_greedy_test36/pred.json",
        "+ clean": B + "results/jobs/ABL_noND_greedy_test36/pred.json", "ECT": B + "results/final/sft_greedy_test36/pred.json"}
ds = L.load("neurips2025", "test"); res = {}
for name, p in RUNS.items():
    preds = json.load(open(p)); ok = stuck = reached = total = 0
    for pr in preds:
        n = len(L.ev.get_ground_truth_data(ds, pr["video_id"])["texts"])
        ins = sum(x.startswith("Instruction") for x in pr["pred_texts"]); suc = sum(x.startswith("Success") for x in pr["pred_texts"])
        ok += ins == min(suc + 1, n); stuck += ins < n; reached += ins; total += n
    res[name] = dict(recordings=len(preds), rule_holds=ok, stopped_early=stuck, steps_reached=reached, steps=total)
    print(name, res[name])
json.dump(res, open(f"{OUT}/advance.json", "w"), indent=1)
