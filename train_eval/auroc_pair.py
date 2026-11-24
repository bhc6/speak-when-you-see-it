import json, random, sys
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    allv = sorted([(v,1) for v in pos]+[(v,0) for v in neg]); r=0; i=0; rs=0
    while i < len(allv):
        j=i
        while j<len(allv) and allv[j][0]==allv[i][0]: j+=1
        avg=(i+j+1)/2; rs+=sum(avg for k in range(i,j) if allv[k][1]); i=j
    return (rs-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def metr(rows):
    m=[r["p_mis"] for r in rows if r["kind"]=="mistake"]; nm=[r["p_mis"] for r in rows if r["kind"]!="mistake"]
    d=[r["p_done"] for r in rows if r["kind"]=="success"]; nd=[r["p_done"] for r in rows if r["kind"]!="success"]
    ms=[r["p_mis"] for r in rows if r["kind"]=="success"]; mz=[r["p_mis"] for r in rows if r["kind"]=="silence"]
    return dict(mis=auc(m,nm), done=auc(d,nd), mis_vs_succ=auc(m,ms), mis_vs_sil=auc(m,mz))
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2]))
key=lambda r:(r["video_id"],r["step"],r["kind"],round(r["t"],2))
bk={key(r):r for r in B}; pairs=[(a,bk[key(a)]) for a in A if key(a) in bk]
print("paired", len(pairs))
vids=sorted({a["video_id"] for a,_ in pairs}); byv={v:[] for v in vids}
for a,b in pairs: byv[a["video_id"]].append((a,b))
ma=metr([a for a,_ in pairs]); mb=metr([b for _,b in pairs])
rng=random.Random(0); diffs={k:[] for k in ma}
for _ in range(500):
    s=[p for v in rng.choices(vids,k=len(vids)) for p in byv[v]]
    x=metr([a for a,_ in s]); y=metr([b for _,b in s])
    for k in diffs: diffs[k].append(y[k]-x[k])
for k in ma:
    d=sorted(diffs[k]); print(f"{k:12s} A {ma[k]:.3f}  B {mb[k]:.3f}  B-A {mb[k]-ma[k]:+.3f} [{d[12]:+.3f},{d[487]:+.3f}]")
