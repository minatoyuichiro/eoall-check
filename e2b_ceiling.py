"""SU(2)-symmetry ceiling on state-preparation fidelity.
Exchange pulses commute with total spin; within a fixed S_z sector the reachable
unitaries are (at most) the direct sum of U(m_S) over total-spin-S blocks.  Hence
    max_U |<t|U|psi0>| = sum_S ||P_S t|| * ||P_S psi0||.
Compare with the optimised fidelities from e2_reach.py."""
import json, sys, math
import numpy as np, torch
from eo_net import *

def spin_ops(N):
    sx = np.array([[0, 1], [1, 0]]) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.array([[1, 0], [0, -1]]) / 2
    I = np.eye(2)
    def op(s, k):
        m = np.array([[1.0]])
        for j in range(N):
            m = np.kron(m, s if j == k else I)
        return m
    Sx = sum(op(sx, k) for k in range(N)); Sy = sum(op(sy, k) for k in range(N)); Sz = sum(op(sz, k) for k in range(N))
    return Sx @ Sx + Sy @ Sy + Sz @ Sz, Sz

def sector_projectors(N):
    S2, Sz = spin_ops(N)
    w, V = np.linalg.eigh(S2 + 1e-3 * Sz)      # split degeneracies by Sz too
    S2w = np.real(np.einsum('ij,ij->j', V.conj(), S2 @ V))
    Svals = np.round((-1 + np.sqrt(1 + 4 * S2w)) / 2, 6)
    Mvals = np.round(np.real(np.einsum('ij,ij->j', V.conj(), Sz @ V)), 6)
    projs = {}
    for S in sorted(set(Svals)):
        for M in sorted(set(Mvals)):
            idx = np.where((Svals == S) & (Mvals == M))[0]
            if len(idx):
                Vs = V[:, idx]; projs[(S, M)] = Vs @ Vs.conj().T
    return projs

def ceiling(psi0, t, projs):
    p0 = psi0.reshape(-1).numpy(); tt = t.reshape(-1).numpy()
    tot = 0.0
    for (S, M), P in projs.items():
        tot += np.linalg.norm(P @ tt) * np.linalg.norm(P @ p0)
    return tot ** 2

def sector_weights(psi, projs):
    v = psi.reshape(-1).numpy()
    return {k: float(np.linalg.norm(P @ v) ** 2) for k, P in projs.items() if np.linalg.norm(P @ v) ** 2 > 1e-9}

if __name__ == '__main__':
    for enc, n in [('st', 3), ('st', 4), ('eo', 2), ('eo', 3)]:
        B, N = logical_basis(enc, n)
        projs = sector_projectors(N)
        gen = torch.Generator().manual_seed(7)
        targets = [('haar', haar_logical(n, gen)) for _ in range(6)] + [('mps2', random_mps_logical(n, 2, gen)) for _ in range(6)]
        targets = [(k, embed_logical(enc, n, c)) for k, c in targets]
        inits = {'st': ['0', '1', 'alt'], 'eo': ['0']}[enc]
        mult = {k: int(round(np.trace(P).real)) for k, P in projs.items()}
        print(f"\n== {enc} n={n} N={N}: sector dims (S,M)->m_S: " + ", ".join(f"({S},{M}):{m}" for (S, M), m in sorted(mult.items()) if abs(M - (0 if enc=='st' else n/2)) < 1e-6))
        for w in inits:
            psi0 = product_state(enc, n, w)
            sw = sector_weights(psi0, projs)
            cs = {'haar': [], 'mps2': []}
            for k, t in targets:
                cs[k].append(ceiling(psi0, t, projs))
            print(f"  init |{w}>: sector weights {{S: w}} = { {S: round(v,3) for (S, M), v in sw.items()} }")
            print(f"     ceiling F_max: haar mean {np.mean(cs['haar']):.4f} (min {np.min(cs['haar']):.4f})   mps2 mean {np.mean(cs['mps2']):.4f} (min {np.min(cs['mps2']):.4f})")
