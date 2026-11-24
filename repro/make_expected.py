"""expected_tables.json for reproduce_tables.py, taken from the outputs of the ORIGINAL analyses (budget_all.json for the same-validation-alarm-cap table,
long_decomp.json for the long-step table), not from reproduce_tables.py itself. python make_expected.py OUT_DIR"""
import json, os, sys
D = "DATA_ROOT/analysis_code"; OUT = sys.argv[1]
bg = json.load(open(f"{D}/budget_clean/out/budget_all.json")); ld = json.load(open(f"{D}/notdone_audit/out/long_decomp.json")); E = {}
for r in bg["rows"]:
    if r["rule"] != "entries": continue
    for k in ("qicd", "mc"):
        x = r[k]; E[f"BUDGET/{k}/{r['variant']}"] = dict(tau=f"{r['tau']:.2f}", ic=x["variant"][0], f1=x["variant"][1], dIC=x["dIC"], dIC95=[round(v, 1) for v in x["dIC95"]],
                                                     dF1=x["dF1"], dF195=[round(v, 3) for v in x["dF195"]], eqIC=x["equiv_IC"], eqF1=x["equiv_F1"])
for b in ("mc", "qicd"):
    for c, cc in (("on time", "on_time"), ("early", "early"), ("alarm", "alarm")):
        E[f"LONG/{b}/{cc}"] = dict(levels={g: ld[b]["levels"][g][c] for g in ("ECT", "+ clean", "+ notdone", "no neg.")}, diff=ld[b]["diff"][c][0], ci=ld[b]["diff"][c][1])
json.dump(E, open(os.path.join(OUT, "expected_tables.json"), "w"), indent=1); print(len(E), "cells")
