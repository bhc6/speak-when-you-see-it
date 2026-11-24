# Text-only shortcut audit of text-counterfactual data (Ego-CoMist, CaptainCook4D part, QICD train videos) vs ours.
# Classifier sees ONLY the instruction string (hashed uni/bigrams + length + 'starts with Now'), recipe-grouped 5-fold.
import sys; sys.path.insert(0, "WORKSPACE/qic/experiments/plan_repr"); import sia_data, os
FR = os.environ["FR"]
for name, kw in (("ours_v3_no_comist", dict(comist=False)), ("v3_with_comist", dict(comist=True))):
    ex, _ = sia_data.build_v3(FR, "train", **kw)
    import collections; src = collections.Counter((e["kind"], e.get("src") or "qicd") for e in ex if e["kind"] in ("mistake", "clean", "silence", "shifted"))
    print(name, "text-only AUROC mistake vs rest %.3f" % sia_data.shortcut_audit(ex), dict(src), flush=True)
ex4, _ = sia_data.build_v4(FR, "train")
print("v4_comist_rewritten+paired", "text-only AUROC %.3f" % sia_data.shortcut_audit(ex4), flush=True)
