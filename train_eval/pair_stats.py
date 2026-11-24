import sys; sys.path.insert(0, "WORKSPACE/qic/experiments/plan_repr")
import sia_data as D, collections, statistics as st
FR="WORKSPACE/qicFRAMES_ROOT"
ex,_=D.build_v3(FR,"train",seed=0,comist=False)
K=collections.defaultdict(list)
for e in ex: K[e["kind"]].append(e)
print({k:len(v) for k,v in K.items()}, "videos", len({e["video_id"] for e in ex}))
def q(L): L=sorted(L); return "med %.0f [%.0f,%.0f]" % (st.median(L), L[len(L)//4], L[3*len(L)//4])
for k in ("success","notdone","mistake","clean","silence"):
    E=[e["t"]-e["start"] for e in K[k]]; R=[e["end"]-e["t"] for e in K[k]]
    print("%-8s elapsed %s | remaining %s | at/after step end %.2f" % (k, q(E), q(R), sum(r<=0 for r in R)/len(R)))
def auc(p,n):
    a=sorted([(x,1) for x in p]+[(x,0) for x in n]); s=0; i=0
    while i<len(a):
        j=i
        while j<len(a) and a[j][0]==a[i][0]: j+=1
        s+=sum((i+j+1)/2 for k in range(i,j) if a[k][1]); i=j
    return (s-len(p)*(len(p)+1)/2)/(len(p)*len(n))
for pos,neg in (("success","notdone"),("mistake","clean")):
    print("%s vs %s: elapsed-only AUROC %.3f | phase-only (closer to step end) AUROC %.3f" % (pos, neg,
      auc([e["t"]-e["start"] for e in K[pos]],[e["t"]-e["start"] for e in K[neg]]),
      auc([-(e["end"]-e["t"]) for e in K[pos]],[-(e["end"]-e["t"]) for e in K[neg]])))
