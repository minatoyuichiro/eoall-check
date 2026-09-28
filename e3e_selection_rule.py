"""Selection rule for PSB readout under SU(2)-symmetric (exchange-only) dynamics (2-spin S-T).
Outcome b (set T of triplet pairs, k=|T|) projects onto states whose total spin lies in
Lam(k) = {0..k} (k != 1) or {1} (k = 1), so Pi_b P_S = 0 for S outside Lam(k) and, for any set K
of Hamming classes,  P(|b| in K) <= sum_{S in Lam(K)} c_S^2,  c_S = ||P_S psi0||.
TV floor >= max_K [ P_teacher(|b| in K) - sum_{S in Lam(K)} c_S^2 ]_+ .  Compare with e3d ceilings
(alt 0.0117, 1 0.1620, 0 0.0528, neel 0 -- all reproduced exactly)."""
import json, numpy as np, torch
from eo_net import *
from e3d_dist_ceiling import sector_bases

teach = json.load(open('results/e3_teachers.json'))
n = 6; N = 12
bases = sector_bases(N, 0.0)
Ss = sorted(bases)
for init in ['0', 'alt', '1', '000100', '010010', '011011', 'neel']:
    psi0 = product_state('st', n, init).reshape(-1)
    c2 = np.array([float((bases[S].conj().T @ psi0).norm() ** 2) for S in Ss])
    cum = np.cumsum(c2)
    line = f"init {init:7s} c_S^2=" + " ".join(f"{x:.3f}" for x in c2)
    for t in ['cat_V8L2', 'begin_V4L3']:
        p = np.array(teach[t]['p']).reshape(-1)
        w = np.array([bin(i).count('1') for i in range(64)])
        # image of Pi_b with |b|=k has total spin S in Lam(k): {0..k} for k != 1, {1} for k == 1
        # (a single triplet pair cannot couple with singlets to S = 0).  For any set K of Hamming
        # classes,  P(|b| in K) <= sum_{S in Lam(K)} c_S^2,  Lam(K) = union of Lam(k).
        import itertools
        Lam = lambda k: {1} if k == 1 else set(range(k + 1))
        floor = 0.0
        for r in range(1, 8):
            for K in itertools.combinations(range(7), r):
                LamK = set().union(*(Lam(k) for k in K))
                bound = sum(c2[S] for S in LamK)
                floor = max(floor, float(sum(p[w == k].sum() for k in K)) - bound)
        line += f" | {t}: TV floor {floor:.4f}"
    print(line)
