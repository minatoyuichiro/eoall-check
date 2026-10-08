"""Selection-rule floors (Eq. tvfloor) for the two-spin encoding at any n, for product initial states
whose sector weights are known in closed form:
  - computational-basis product states (Neel): <x|P_S|x> = m_S / C(N, N/2)   (permutation symmetry)
  - products of singlets and T0: weights computed by coupling (CG) recursively over pairs.
usage: python3 e10_floors.py teacher1,teacher2 [inits]"""
import json, sys, itertools, math
import numpy as np
from math import comb

def mult(N, S):                       # multiplicity of total spin S at S_z=0, N even
    return comb(N, N // 2 - S) - (comb(N, N // 2 - S - 1) if N // 2 - S - 1 >= 0 else 0)

def cg_couple(dist, j2):
    """couple a spin-j distribution {J: w} with one spin j2 (m=0 for both) -> new {J': w'} using
    |<j 0; j2 0 | J' 0>|^2 weights (computed numerically via sympy-free recursion is overkill: use Wigner 3j via
    scipy)."""
    from sympy.physics.quantum.cg import CG
    from sympy import S as Sy
    out = {}
    for J, w in dist.items():
        for Jp in range(abs(J - j2), J + j2 + 1):
            c = float(CG(Sy(J), 0, Sy(j2), 0, Sy(Jp), 0).doit()) ** 2
            if c > 1e-15: out[Jp] = out.get(Jp, 0.0) + w * c
    return out

def sector_weights(init, n):
    N = 2 * n
    if init == 'neel':
        dim = comb(N, N // 2)
        return {S: mult(N, S) / dim for S in range(n + 1)}
    pat = {'0': [0] * n, '1': [1] * n, 'alt': [j % 2 for j in range(n)]}.get(init)
    if pat is None: pat = [int(ch) for ch in init]
    dist = {0: 1.0}
    for b in pat:
        dist = cg_couple(dist, 0 if b == 0 else 1)
    return dist

def floor(p, n, c2):
    w = np.array([bin(i).count('1') for i in range(2 ** n)])
    mass = np.array([p[w == k].sum() for k in range(n + 1)])
    Lam = lambda k: {1} if k == 1 else set(range(k + 1))
    best = 0.0
    for r in range(1, n + 2):
        for K in itertools.combinations(range(n + 1), r):
            LamK = set().union(*(Lam(k) for k in K))
            best = max(best, mass[list(K)].sum() - sum(c2.get(S, 0.0) for S in LamK))
    return best

if __name__ == '__main__':
    teach = json.load(open('results/e3_teachers.json'))
    names = sys.argv[1].split(','); inits = sys.argv[2].split(',') if len(sys.argv) > 2 else ['neel', 'alt', '0']
    for t in names:
        p = np.array(teach[t]['p']); p = p / p.sum(); n = int(math.log2(len(p)))
        for init in inits:
            c2 = sector_weights(init, n)
            print(f"{t:12s} n={n} init {init:5s} floor(TV) {floor(p, n, c2):.4f}  c_S^2={[round(c2.get(S,0),3) for S in range(n+1)]}", flush=True)
