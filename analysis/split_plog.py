"""Split a shared plog.jsonl (two shards appended concurrently) into per-shard logs of mistake-'yes' records, by matching
each yes record's instruction to the next pending Feedback of each shard (order within a shard is preserved).
Exact iff no step where both shards' next pending Feedback have the same instruction prefix (counted & reported).
Writes plog_yes{0,1}.jsonl next to the shards."""
import json, sys, glob, os
d = sys.argv[1]; S = [json.load(open(f)) for f in sorted(glob.glob(d + '/shard*.json'))]
pend = []
for Sk in S:
    q = []
    for p in Sk:
        ins = None
        for t in p['pred_texts']:
            if t.startswith('Instruction:'): ins = t[len('Instruction:'):].strip()
            elif t.startswith('Feedback'): q.append(ins[:80])
    pend.append(q)
L = [json.loads(l) for l in open(d + '/plog.jsonl')]; Y = [r for r in L if r['mode'] == 'mistake_inference' and r['yes']]
ptr = [0] * len(S); out = [[] for _ in S]; amb = 0; miss = 0
for r in Y:
    c = [k for k in range(len(S)) if ptr[k] < len(pend[k]) and pend[k][ptr[k]] == r['instr'][:80]]
    if not c: miss += 1; continue
    if len(c) > 1: amb += 1
    k = c[0]; out[k].append(r); ptr[k] += 1
ok = miss == 0 and all(ptr[k] == len(pend[k]) for k in range(len(S)))
print(d, 'yes', len(Y), 'feedback', [len(q) for q in pend], 'ambiguous', amb, 'unmatched', miss, 'OK' if ok and amb == 0 else 'NOT_EXACT')
if ok and amb == 0:
    for k in range(len(S)):
        with open(f'{d}/plog_yes{k}.jsonl', 'w') as f:
            for r in out[k]: f.write(json.dumps(r) + '\n')
