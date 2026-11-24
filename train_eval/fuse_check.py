import json, math, random, sys
sys.path.insert(0, "/tmp"); 
exec(open("TMP_PATH").read().split("A=json.load")[0])   # reuse auc()
L = lambda p: math.log(max(p,1e-6)/max(1-p,1e-6))
T = json.load(open("auroc_r5/T1v4_s2400.json")); P = json.load(open("auroc_r5/P1_s1200.json"))
key = lambda r: (r["video_id"], r["step"], r["kind"], round(r["t"], 2))
pk = {key(r): r for r in P}; rows = [(a, pk[key(a)]) for a in T if key(a) in pk]
vids = sorted({a["video_id"] for a, _ in rows}); rng = random.Random(0); rng.shuffle(vids); fold = {v: i % 5 for i, v in enumerate(vids)}
def feats(a, b): return [L(a["p_mis"]), L(b["p_mis"]), L(a["p_done"])]
def fit(X, y, it=300, lr=0.1):
    w = [0.0]*(len(X[0])+1)
    for _ in range(it):
        g = [0.0]*len(w)
        for x, t in zip(X, y):
            z = w[-1] + sum(wi*xi for wi, xi in zip(w, x)); p = 1/(1+math.exp(-max(-30,min(30,z))))
            for i, xi in enumerate(x): g[i] += (p-t)*xi
            g[-1] += p-t
        w = [wi - lr*gi/len(X) for wi, gi in zip(w, g)]
    return w
score = {}
for f in range(5):
    tr = [(a, b) for a, b in rows if fold[a["video_id"]] != f]
    w = fit([feats(a, b) for a, b in tr], [a["kind"] == "mistake" for a, b in tr])
    for a, b in rows:
        if fold[a["video_id"]] == f: score[key(a)] = w[-1] + sum(wi*xi for wi, xi in zip(w, feats(a, b)))
def rep(name, s):
    m=[s(a,b) for a,b in rows if a["kind"]=="mistake"]; n=[s(a,b) for a,b in rows if a["kind"]!="mistake"]
    su=[s(a,b) for a,b in rows if a["kind"]=="success"]; si=[s(a,b) for a,b in rows if a["kind"]=="silence"]
    print(f"{name:28s} mis {auc(m,n):.3f}  mis_vs_succ {auc(m,su):.3f}  mis_vs_sil {auc(m,si):.3f}")
rep("T1v4@2400", lambda a,b: a["p_mis"]); rep("P1@1200", lambda a,b: b["p_mis"])
rep("sum of logits (no fit)", lambda a,b: L(a["p_mis"])+L(b["p_mis"]))
rep("CV logistic (T,P,T_done)", lambda a,b: score[key(a)])
