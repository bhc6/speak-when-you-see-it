"""Offline replay of the per-step protocol from qic_dense.py logs (QICD test36), CPU only.
Same loop as train_eval/dense_sim.py (done_first: completion question first, Feedback only if the step is not confirmed;
an unconfirmed step runs to the end of the video), scored with the local official eval. A replay is exact if it never
needs a check beyond the logged ones: with --stop 0.9 --mis_lo 0.3 every done_first policy with tau_done <= 0.9 ends at
or before the logged stop; steps cut by the 60 s cap are counted ('caps') and such rows are flagged inexact.
  1 validation   : ECT s0 at the frozen thresholds (0.2, 0.35) must equal its official probability run (42.0 / 0.306).
  2 sweep        : tau_done x tau_mis grid for ECT s0 and no paired neg. s0 (IC, F1, margin over the blind frontier).
  3 hybrid       : completion probabilities of one model with the mistake probabilities of the other, on the checks both
                   logged (a step that needs a check only one model logged is counted as inexact) -> which head carries
                   the gain, at the frozen thresholds and at each model's best tau_mis.
  4 mechanism    : per mistake-free step, the number k of checks before the replay confirms it and the per-check alarm
                   rate p; compare the share of steps voided with 1 - mean((1 - p)^k).
python3 replay.py -> out/replay.json"""
import sys, os, json, types, collections, statistics as st
_t = types.ModuleType("torch"); _t.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
sys.modules.update({"torch": _t, "torch.utils": _t.utils, "torch.utils.data": _t.utils.data})
V0 = "DATA_ROOT/"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V0 + "analysis_code/blind_strong"); sys.path.insert(0, V0 + "analysis_code")
import local_eval as L
from frontier_canon import fr
SRC = os.environ.get("CLK_DIR", V0 + "machine5/results/clk1122")
OUT = HERE + "/out"; os.makedirs(OUT, exist_ok=True)
DS = L.load("neurips2025", "test"); GT = {a["video_id"]: a for a in DS.annotations}
FROZEN = (0.2, 0.35)


def load(name):
    D = {}
    for i in (0, 1):
        for l in open(f"{SRC}/dense_{name}.shard{i}.jsonl"):
            if l.strip(): r = json.loads(l); D[r["video_id"]] = r
    return D


def replay(VD, VM, td, tm, cont=True):
    """VD gives completion probs, VM mistake probs (same model -> VD is VM). Returns preds, inexact-step count, per-step log.
    cont: a step still unconfirmed at the 60 s cap is taken as confirmed at the next check with no further alarm, and the
    replay continues (the next step restarts at its annotated start). For IC this only misses the case where the real run
    never confirms it before the video ends; its alarms after end + 60 s are dropped. Such steps are counted as inexact."""
    preds, inexact, steps = [], 0, []
    for vid in sorted(VD):
        vd, vm = VD[vid], VM[vid]; texts, ts = [], []
        for sd, sm in zip(vd["steps"], vm["steps"]):
            assert sd["k"] == sm["k"] and abs(sd["start"] - sm["start"]) < 1e-6
            texts.append("Instruction: " + sd["instruction"]); ts.append(sd["start"])
            cm = {c[0]: c for c in sm["checks"]}; fin = False; last = None; k = 0; fired = 0; ok = True
            for c in sd["checks"]:
                seek, t, pd = c[0], c[1], c[2]
                if vd is not vm and seek not in cm: ok = False; break
                pm = c[3] if vd is vm else cm[seek][3]
                last = seek; k += 1
                if pd >= td: texts.append("Success: Good job!"); ts.append(t); fin = True; break
                if pm >= tm: texts.append("Feedback: you made a mistake."); ts.append(t); fired += 1
            steps.append(dict(video=vid, k_step=sd["k"], checks=k, alarms=fired, confirmed=fin))
            if not ok: inexact += 1; break
            if not fin:
                capped = sd["why"] in ("cap", "stop") or (vd is not vm and sm["why"] in ("cap", "stop"))
                inexact += capped
                if not (cont and capped and last is not None): break
                texts.append("Success: Good job!"); ts.append(sd["checks"][-1][1] + 5.0)
            if last + 10 >= vd["n_frames"]: break
        preds.append({"video_id": vid, "pred_texts": texts, "pred_timestamps": ts})
    return preds, inexact, steps


def score(preds):
    s, _ = L.run(DS, preds); s["margin"] = round(s["f1"] - fr("qicd", s["ic"]), 4); return s


res = {}
E, N = load("ECT_s0"), load("noneg_s0")
assert len(E) == 36 and len(N) == 36, (len(E), len(N))
# 1 validation
p, inx, _ = replay(E, E, *FROZEN); s = score(p)
res["validation_ECT_s0_frozen"] = dict(inexact=inx, **s, official=dict(ic=42.0, f1=0.306))
print("validation ECT s0 frozen", s, "inexact", inx, "(official prob run 42.0 / 0.306)")
for nm, D in (("ECT_s0", E), ("noneg_s0", N)):   # the same replay ending each recording at a step that outlasts the log
    p0, inx0, _ = replay(D, D, *FROZEN, cont=False); res[f"no_continuation_{nm}"] = dict(inexact=inx0, **score(p0))
    print("without cap continuation", nm, res[f"no_continuation_{nm}"]["ic"])
pn, inx, _ = replay(N, N, *FROZEN); s = score(pn)
res["validation_noneg_s0_frozen"] = dict(inexact=inx, **s, official=dict(ic=31.1, f1=0.244))
print("validation noneg s0 frozen", s, "inexact", inx, "(official prob run 31.1 / 0.244)")
# 2 sweep
TD = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; TM = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]
for name, D in (("ECT s0", E), ("no paired neg. s0", N)):
    rows = []
    for td in TD:
        for tm in TM:
            p, inx, _ = replay(D, D, td, tm); rows.append(dict(tau_done=td, tau_mis=tm, inexact=inx, **score(p)))
    res[f"sweep|{name}"] = rows
    na = [r for r in rows if r["tau_mis"] == 1.01]; best = max(rows, key=lambda r: r["margin"])
    print(f"{name}: IC without alarms by tau_done", [(r["tau_done"], r["ic"]) for r in na])
    print(f"   best margin {best['margin']:+.3f} at {best['tau_done']}/{best['tau_mis']} (IC {best['ic']}, F1 {best['f1']}); "
          "best F1 at IC>=40/45/50:", [max([r["f1"] for r in rows if r["ic"] >= x], default=None) for x in (40, 45, 50)],
          "(all rows have 10-31 inexact steps: cap continuation, see replay())")
# 3 hybrid
for lab, VD, VM in (("completion ECT + mistake noneg", E, N), ("completion noneg + mistake ECT", N, E)):
    rows = []
    for tm in [FROZEN[1]] + TM[1:]:
        p, inx, _ = replay(VD, VM, FROZEN[0], tm); rows.append(dict(tau_done=FROZEN[0], tau_mis=tm, inexact=inx, **score(p)))
    res[f"hybrid|{lab}"] = rows
    print(lab, [(r["tau_mis"], r["ic"], r["f1"], r["inexact"]) for r in rows])
# 3b the same exchange at thresholds of 0.5 on both questions, the closest to greedy decoding (more steps outlast the log here)
for lab, VD, VM in (("ECT", E, E), ("no paired neg.", N, N), ("completion noneg + mistake ECT", N, E)):
    p, inx, _ = replay(VD, VM, 0.5, 0.5); res[f"near_greedy|{lab}"] = dict(inexact=inx, **score(p))
    print("near greedy (0.5/0.5)", lab, res[f"near_greedy|{lab}"]["ic"], "inexact", inx)
# 4 mechanism at the frozen thresholds
for name, D in (("ECT s0", E), ("no paired neg. s0", N)):
    _, _, steps = replay(D, D, *FROZEN); clean = []
    for s in steps:
        if not GT[s["video"]]["instrcution_segment_has_mistake"][s["k_step"]]: clean.append(s)
    al = sum(s["alarms"] for s in clean); ch = sum(s["checks"] for s in clean); p = al / max(ch, 1)
    pred = 1 - st.mean((1 - p) ** s["checks"] for s in clean); act = sum(s["alarms"] > 0 for s in clean) / len(clean)
    res[f"mechanism|{name}"] = dict(clean_steps_reached=len(clean), checks=ch, per_check_alarm_rate=round(p, 4),
                                    mean_checks=round(ch / len(clean), 2), voided_actual=round(act, 3), voided_predicted=round(pred, 3))
    print(f"{name}: clean steps reached {len(clean)}, checks/step {ch / len(clean):.2f}, per-check alarm rate {p:.3f}, "
          f"voided actual {act:.3f} vs 1-(1-p)^k {pred:.3f}")
json.dump(res, open(OUT + "/replay.json", "w"), indent=1)
