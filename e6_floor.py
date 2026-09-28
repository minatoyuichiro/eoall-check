"""Selection-rule floors (Eq. tvfloor) for the 8-qubit teachers, st n=8 (N=16), using the S_z=0
sector and polynomial projectors P_S = prod_{S'!=S} (S^2 - S'(S'+1)) / (S(S+1) - S'(S'+1))."""
import json, itertools, numpy as np, torch
from eo_sector import SectorSim
n = 8; S = SectorSim('st', n); N = S.N
teach = json.load(open('results/e3_teachers.json'))
def S2(v):                        # S^2 = 3N/4 + sum_{i<j} (SWAP_ij - 1/2)
    out = (3 * N / 4 - N * (N - 1) / 4) * v
    for i in range(N):
        for j in range(i + 1, N):
            mi, mj = 1 << (N - 1 - i), 1 << (N - 1 - j)
            bi = (S.basis & mi) != 0; bj = (S.basis & mj) != 0
            y = np.where(bi != bj, S.basis ^ (mi | mj), S.basis)
            out = out + v[S.pos[y]]
    return out
Ss = list(range(n + 1))
Lam = lambda k: {1} if k == 1 else set(range(k + 1))
for init in ['alt', 'neel', '0', '1']:
    v0 = S.product_state(init).numpy()
    c2 = []
    for s in Ss:
        v = v0.copy()
        for t in Ss:
            if t != s: v = (S2(v) - t * (t + 1) * v) / (s * (s + 1) - t * (t + 1))
        c2.append(float(np.vdot(v0, v).real))
    c2 = np.array(c2)
    line = f"init {init:5s} c_S^2=" + " ".join(f"{x:.3f}" for x in c2) + f" (sum {c2.sum():.3f})"
    for t in ['cat_V16L2', 'begin_V4L4']:
        p = np.array(teach[t]['p']).reshape(-1); w = np.array([bin(i).count('1') for i in range(2 ** n)])
        floor = 0.0
        for r in range(1, n + 2):
            for K in itertools.combinations(range(n + 1), r):
                LamK = set().union(*(Lam(k) for k in K)); bound = sum(c2[s] for s in LamK)
                floor = max(floor, float(sum(p[w == k].sum() for k in K)) - bound)
        line += f" | {t}: floor {floor:.4f} (p|b|=1: {p[w==1].sum():.4f}, p0: {p[0]:.4f})"
    print(line, flush=True)
