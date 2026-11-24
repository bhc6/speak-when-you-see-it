"""Run the official QICD/Ego-MC eval.py locally on annotations dumped from box8 (inputs/gt_dump_1118.json, md5 d6f95b06;
dumped with utils.load_dataset(dv, "advanced_planning", split) -> ds.annotations, i.e. after the dataset classes' __init__).
Stubs: bert_score (fluency only, not used by IC/F1), datasets (annotations come from the dump), rouge_score (ROUGE-L F on
lowercased [a-z0-9]+ tokens; only used to check the instruction, which every run here copies verbatim from the plan).
The stub is validated against official eval.log numbers in check_local_eval() before any new result is produced."""
import sys, os, io, re, json, types, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); EV = os.path.join(HERE, "inputs", "official_eval")
def _lcs(a, b):
    p = [0] * (len(b) + 1)
    for x in a:
        q = [0]
        for j, y in enumerate(b): q.append(p[j] + 1 if x == y else max(p[j + 1], q[j]))
        p = q
    return p[-1]
class _Rouge:
    def __init__(self, *a, **k): pass
    def score(self, pred, gt):
        a, b = re.findall(r"[a-z0-9]+", pred.lower()), re.findall(r"[a-z0-9]+", gt.lower())
        l = _lcs(a, b); f = 0.0 if not l else 2 * l / (len(a) + len(b))
        return {"rougeL": types.SimpleNamespace(fmeasure=f)}
_bs = types.ModuleType("bert_score"); _bs.BERTScorer = lambda *a, **k: None; sys.modules["bert_score"] = _bs
_rs = types.ModuleType("rouge_score"); _rs.rouge_scorer = types.SimpleNamespace(RougeScorer=_Rouge); sys.modules["rouge_score"] = _rs
_ds = types.ModuleType("datasets"); _ds.load_dataset = None; sys.modules["datasets"] = _ds
sys.path.insert(0, EV)
import data as _data, eval as ev  # noqa: E402
ev.get_bert_score = lambda *x: 0.0
_DUMP = None
def load(dv, sp):
    global _DUMP
    if _DUMP is None: _DUMP = json.load(open(os.path.join(HERE, "inputs", os.environ.get("QIC_GT_DUMP", "gt_dump_1118.json"))))
    cls = _data.QualcommInteractiveCookingDatasetNeurIPS2025 if dv == "neurips2025" else _data.EgoMCBench
    ds = cls.__new__(cls); ds.annotations = _DUMP[f"{dv}_{sp}"]; return ds
def run(ds, preds):
    """Official run(); returns the last cumulative block (IC, tp, fp, tn, fn, F1) and the raw output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()): ev.run(ds, preds)
    o = buf.getvalue(); ic = float(re.findall(r"IC-Acc: ([\d.]+)", o)[-1])
    tp, fp, tn, fn = map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1])
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-9)
    return dict(ic=ic, tp=tp, fp=fp, tn=tn, fn=fn, f1=round(f1, 4)), o
