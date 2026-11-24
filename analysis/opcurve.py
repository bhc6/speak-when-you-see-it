"""Operating curves from existing qic_prob runs (exact offline; done_first => raising tau_mis only removes Feedback).
For every finished run dir with shard*.json + plog*.jsonl: tau_mis in {logged, .4, .5, ..., .95, .99, 1.01} -> official eval
(IC, TP/FP/TN/FN, #feedback). Output: WORKSPACE/qic/results/opcurve/<name>.json"""
import sys, os, json, glob, io, contextlib, re
EV='WORKSPACE/qic/ref_sys/qualcomm_interactive_cooking_eval'; sys.path.insert(0, EV); os.chdir(EV)
import eval as ev, utils
ev.get_bert_score = lambda *x: 0.0
OUT='WORKSPACE/qic/results/opcurve'; os.makedirs(OUT, exist_ok=True); DS={}
def score(dv, sp, P):
    if (dv,sp) not in DS:
        with contextlib.redirect_stdout(io.StringIO()): DS[(dv,sp)] = utils.load_dataset(dv, "advanced_planning", sp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ev.run(DS[(dv,sp)], P)
    o = buf.getvalue(); ic = float(re.findall(r"IC-Acc: ([\d.]+)", o)[-1])
    tp, fp, tn, fn = map(int, re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", o)[-1])
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); return dict(ic=ic, tp=tp, fp=fp, tn=tn, fn=fn, f1=round(2*p*r/max(p+r,1e-9),4))
dirs = [d for d in glob.glob('WORKSPACE/qic/results/**/*prob*', recursive=True) if os.path.isdir(d) and os.path.exists(d+'/DONE') and glob.glob(d+'/plog*.jsonl') and '_discarded' not in d]
for d in sorted(dirs):
    name = d.replace('WORKSPACE/qic/results/','').replace('/','__')
    if os.path.exists(f'{OUT}/{name}.json'): continue
    n = d.lower(); dv = 'ego_mc_bench' if '_mc' in n else 'neurips2025'; sp = 'validation' if '_val' in n else 'test'
    shards = sorted(glob.glob(d+'/shard*.json')); plogs = sorted(glob.glob(d+'/plog_yes*.jsonl')) or sorted(glob.glob(d+'/plog[0-9]*.jsonl')) or sorted(glob.glob(d+'/plog.jsonl'))
    if len(plogs) == 1 and len(shards) > 1: print('skip (single plog for many shards)', name); continue
    if len(shards) != len(plogs): print('skip (shard/plog mismatch)', name); continue
    S = [json.load(open(s)) for s in shards]; L = [[json.loads(l) for l in open(p)] for p in plogs]
    if any(r.get('mode') != 'mistake_inference' for Lk in L for r in Lk if False): pass
    if any(r.get('order','done_first')!='done_first' for Lk in L for r in Lk): print('skip order', name); continue
    tau0 = max(r['tau'] for Lk in L for r in Lk if r['mode']=='mistake_inference')
    ok = all(sum(t.startswith('Feedback') for p in Sk for t in p['pred_texts']) == sum(r['mode']=='mistake_inference' and r['yes'] for r in Lk) for Sk, Lk in zip(S, L))
    if not ok: print('skip (feedback/log mismatch)', name); continue
    rows = []
    for tau in [tau0] + [x for x in (.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99,1.01) if x > tau0]:
        P = []; nfb = 0
        for Sk, Lk in zip(S, L):
            it = iter([r for r in Lk if r['mode']=='mistake_inference' and r['yes']])
            for p in Sk:
                T, TS = [], []
                for t, ts in zip(p['pred_texts'], p['pred_timestamps']):
                    if t.startswith('Feedback'):
                        if next(it)['p'] < tau: continue
                        nfb += 1
                    T.append(t); TS.append(ts)
                P.append({'video_id': p['video_id'], 'pred_texts': T, 'pred_timestamps': TS})
        assert len({p['video_id'] for p in P}) == len(P)
        rows.append(dict(tau=tau, n_feedback=nfb, **score(dv, sp, P)))
    json.dump({'run': name, 'dataset': dv, 'split': sp, 'tau_done': L[0][0]['tau'] if L[0][0]['mode']=='instruction_end' else None, 'curve': rows}, open(f'{OUT}/{name}.json','w'), indent=0)
    print('done', name, len(rows), 'points', flush=True)
print('OPCURVE_DONE')
