"""How much of the duration prior the training pairs leave (supplement H). Re-simulates the paired negatives of ECT from the QICD
training annotations with the rule of train_eval/sia_data.build_v3 (success situations at the step end and 5 s later with a
partner step of the same recording lasting > E + 20 s; mistake situations at the feedback time and 5 and 10 s later with a
mistake-free partner step lasting > E + 15 s), which gives the same 4,068 and 1,159 pairs as the training set. Within each
pair, ranks the two sides by elapsed time over the median duration of their instruction in the other training videos
(leave-one-video-out); ties count one half. Three random draws of the partners.
python3 train_pair_prior.py -> out/train_pair_prior.json"""
import json, statistics as st, collections, random, bisect
V="DATA_ROOT/"; import os; HERE=os.path.dirname(os.path.abspath(__file__)); OUT={}
g=json.load(open(V+"analysis_code/blind_strong/inputs/gt_dump_1118.json"))
val={v['video_id'] for v in g['neurips2025_validation']}
rows=[v for v in g['neurips2025_train'] if v['video_id'] not in val]
norm=lambda s: s.replace("Instruction:","").strip()
dur=collections.defaultdict(list)  # instr -> list of (vid, d)
for r in rows:
    for tx,ts in zip(r['instruction_segment_texts'],r['instruction_segment_texts_timestamps']):
        if max(ts)>ts[0]: dur[norm(tx[0])].append((r['video_id'],max(ts)-ts[0]))
GM=st.median([d for x in dur.values() for _,d in x])
def prior(ins,vid):
    x=[d for v,d in dur.get(ins,[]) if v!=vid]
    return st.median(x) if x else GM
ex=[]
for r in rows:
    vid=r['video_id']
    for si,(tx,ts,ty,hm) in enumerate(zip(r['instruction_segment_texts'],r['instruction_segment_texts_timestamps'],r['instruction_segment_texts_types'],r['instrcution_segment_has_mistake'])):
        start,end=ts[0],max(ts); ins=norm(tx[0])
        fb=[t for t,y in zip(ts[1:],ty[1:]) if 'feedback' in y]
        mis=[t for t,y in zip(ts[1:],ty[1:]) if 'mistake' in y]
        base=dict(video_id=vid,step=si,start=start,end=end,ins=ins)
        for f in mis:
            for dt in (0,5,10):
                if f+dt>=start+5: ex.append(dict(base,kind='mistake',t=f+dt))
        if not hm:
            for dt in (0,5):
                if end+dt>=start+5: ex.append(dict(base,kind='success',t=end+dt))
        t=start+10
        while t<=end-15:
            if all(abs(t-f)>=15 for f in fb): ex.append(dict(base,kind='silence',t=t))
            t+=20
steps=collections.defaultdict(set)
for e in ex: steps[e['video_id']].add((e['step'],e['start'],e['end'],e['ins']))
misst={(e['video_id'],e['step']) for e in ex if e['kind']=='mistake'}
mfree=collections.defaultdict(set)
for e in ex:
    if e['kind'] in ('silence','success'): mfree[e['video_id']].add((e['step'],e['start'],e['end'],e['ins']))
def auc(p,n):
    n=sorted(n); return sum(bisect.bisect_left(n,x)+0.5*(bisect.bisect_right(n,x)-bisect.bisect_left(n,x)) for x in p)/max(len(p)*len(n),1)
for seed in (7,8,9):
  rng=random.Random(seed)
  for kind,gap,pool in (('success',20,steps),('mistake',15,mfree)):
    P=[];N=[];Pp=[];Np=[];acc=[]
    for e in ex:
        if e['kind']!=kind: continue
        E=e['t']-e['start']
        cand=[s for s in sorted(pool[e['video_id']]) if s[0]!=e['step'] and s[2]-s[1]>E+gap and (kind=='success' or (e['video_id'],s[0]) not in misst)]
        P.append(E); a=E/prior(e['ins'],e['video_id']); Pp.append(a)
        if cand:
            s=rng.choice(cand); N.append(E); b=E/prior(s[3],e['video_id']); Np.append(b)
            acc.append(1 if a>b else 0.5 if a==b else 0)
    OUT[f"{kind}|draw{seed}"]=dict(pairs=len(acc), raw_elapsed_auroc=round(auc(P,N),3), prior_relative_auroc=round(auc(Pp,Np),3), within_pair_accuracy=round(sum(acc)/len(acc),3))
    print(seed,kind,OUT[f"{kind}|draw{seed}"])
json.dump(OUT, open(HERE+"/out/train_pair_prior.json","w"), indent=1)
