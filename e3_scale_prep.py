"""E6 prep: 256-outcome teachers (8 logical qubits) and their MPS fits, appended to the E3 json files.
cat_V16L2: 'The cat sat on', top-16 x 2 tokens; begin_V4L4: 'In the beginning', top-4 x 4 tokens."""
import json, math, sys
import numpy as np
sys.argv = sys.argv[:1]
import e3_teacher as T
import e3_mps as M

teachers = json.load(open('results/e3_teachers.json'))
for name, prompt, V, L in [('cat_V16L2', 'The cat sat on', 16, 2), ('begin_V4L4', 'In the beginning', 4, 4)]:
    if name in teachers: continue
    words, P = T.teacher(prompt, V, L)
    H = -(P * np.log2(P + 1e-30)).sum()
    print(f"{name}: vocab={words}  H={H:.3f} bits (max {math.log2(len(P)):.1f})  max p={P.max():.3f}", flush=True)
    teachers[name] = dict(prompt=prompt, V=V, L=L, vocab=words, p=P.tolist())
json.dump(teachers, open('results/e3_teachers.json', 'w'), indent=1)

mps = json.load(open('results/e3_mps.json'))
for name in ['cat_V16L2', 'begin_V4L4']:
    d = teachers[name]; n = int(math.log2(len(d['p'])))
    mps.setdefault(name, {})
    for chi in [1, 2, 4, 8, 16]:
        if f'chi{chi}' in mps[name]: continue
        kls = []; cs = []
        for seed in range(3):
            kl, c = M.fit(d['p'], n, chi, seed=seed); kls.append(kl); cs.append(c)
        k = int(np.argmin(kls)); c = cs[k]
        mps[name][f'chi{chi}'] = dict(kl=kls[k], kl_seeds=kls, amp=c.numpy().tolist(), p=(c.numpy() ** 2).tolist())
        print(f"{name} chi={chi}: KL(teacher||MPS) = {kls[k]:.4f} nats  (seeds {np.round(kls,4)})", flush=True)
        json.dump(mps, open('results/e3_mps.json', 'w'))
