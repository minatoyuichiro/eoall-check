"""Sector-restricted state-vector simulator for the exchange networks of eo_net.py.

Exchange pulses conserve S_z, so a chain started in a product state with a fixed number of
down spins never leaves that sector: C(N, n_down) amplitudes instead of 2^N
(eo n=8: 735 471 instead of 16 777 216).  Same conventions as eo_net (index bit 0 = up,
site 0 = most significant bit); pulse on pair (i,i+1):
    psi' = e^{i a/4} [cos(a/2) psi - i sin(a/2) psi[perm_pair]]
where perm_pair swaps the two bits (identity where they are equal).
"""
import math
import numpy as np, torch
import scipy.sparse as sp
from eo_net import CDT, RDT, unit_states, n_params

class SectorSim:
    def __init__(self, enc, n):
        self.enc, self.n = enc, n
        self.b = 2 if enc == 'st' else 3
        self.N = N = self.b * n
        n_down = n                                        # st: S_z=0 ; eo: S_z=n/2 (2n up, n down)
        x = np.arange(2 ** N, dtype=np.int64)
        pc = np.zeros(2 ** N, dtype=np.int8)
        for i in range(N): pc += ((x >> i) & 1).astype(np.int8)
        self.basis = x[pc == n_down]                      # sorted integers
        self.D = len(self.basis)
        pos = np.full(2 ** N, -1, dtype=np.int64); pos[self.basis] = np.arange(self.D)
        self.pos = pos
        self._perm = {}
        # sparse code basis (2^n x D), real
        bs, units = unit_states(enc)
        ent = []
        for u in units:                                   # list of (bits tuple, coef)
            uu = u.numpy().real
            ent.append([(idx, float(uu[idx])) for idx in zip(*np.nonzero(uu))])
        rows, cols, vals = [], [], []
        import itertools
        for L in range(2 ** n):
            lb = [(L >> (n - 1 - j)) & 1 for j in range(n)]
            for combo in itertools.product(*[ent[lb[j]] for j in range(n)]):
                bits = sum([list(c[0]) for c in combo], [])
                xi = int(''.join(map(str, bits)), 2)
                coef = float(np.prod([c[1] for c in combo]))
                rows.append(L); cols.append(pos[xi]); vals.append(coef)
        assert min(cols) >= 0
        self.B = sp.csr_matrix((vals, (rows, cols)), shape=(2 ** n, self.D))
        # readout bookkeeping for the first pair of every block
        self.ro_perm, self.ro_sign, self.ro_diff = [], [], []
        key = np.zeros(self.D, dtype=np.int64)
        for j in range(n):
            i = self.b * j
            perm, diff, s = self._pair(i)
            self.ro_perm.append(perm); self.ro_diff.append(diff); self.ro_sign.append(s)
            bi = (self.basis >> (N - 1 - i)) & 1; bj = (self.basis >> (N - 2 - i)) & 1
            singlet_slot = (bi == 0) & (bj == 1)          # |up down> slot holds the singlet amplitude after the transform
            key += (~singlet_slot).astype(np.int64) << (n - 1 - j)
        self.key = torch.tensor(key)

    def _pair(self, i):
        if i in self._perm: return self._perm[i]
        N = self.N
        mi, mj = 1 << (N - 1 - i), 1 << (N - 2 - i)
        bi = (self.basis & mi) != 0; bj = (self.basis & mj) != 0
        diff = bi != bj
        y = np.where(diff, self.basis ^ (mi | mj), self.basis)
        perm = torch.tensor(self.pos[y])
        sign = torch.tensor(np.where(bi, 1.0, -1.0), dtype=RDT)   # (1,0)=|down up|: +, (0,1)=|up down|: -
        self._perm[i] = (perm, torch.tensor(diff), sign)
        return self._perm[i]

    # -------------------------------------------------------------- states
    def embed(self, c):
        """logical amplitudes (2^n,) -> sector vector (D,)"""
        c = np.asarray(c, dtype=np.complex128)
        v = self.B.T @ c
        return torch.tensor(v, dtype=CDT)

    def product_state(self, which='0'):
        n = self.n
        if which == 'neel':
            c = np.ones(2 ** n) / math.sqrt(2 ** n)
        else:
            pat = {'0': [0] * n, '1': [1] * n, 'alt': [j % 2 for j in range(n)]}.get(which)
            if pat is None: pat = [int(ch) for ch in which]
            c = np.zeros(2 ** n); c[int(''.join(map(str, pat)), 2)] = 1.0
        return self.embed(c)

    # -------------------------------------------------------------- dynamics
    def pulse(self, psi, i, a):
        perm, diff, _ = self._pair(i)
        a = a.to(RDT)
        return torch.exp(1j * a / 4) * (torch.cos(a / 2) * psi - 1j * torch.sin(a / 2) * psi[perm])

    def run(self, psi0, thetas, sched):
        psi = psi0; k = 0
        for rnd in sched:
            for (i, j) in rnd:
                psi = self.pulse(psi, i, thetas[k]); k += 1
        return psi

    # -------------------------------------------------------------- readout
    def psb_probs(self, psi):
        x = psi
        for j in range(self.n):
            perm, diff, s = self.ro_perm[j], self.ro_diff[j], self.ro_sign[j]
            x = torch.where(diff, (x + s * x[perm]) / math.sqrt(2), x)
        q = x.abs() ** 2
        p = torch.zeros(2 ** self.n, dtype=RDT).index_add(0, self.key, q)
        return p.reshape((2,) * self.n)

    def leakage(self, psi):
        v = psi.detach().numpy()
        amp = self.B @ v
        return 1.0 - float((np.abs(amp) ** 2).sum())

    def fidelity(self, psi, target):
        return (torch.vdot(target, psi).abs() ** 2).real


if __name__ == '__main__':          # check against the dense simulator
    from eo_net import bubble_schedule, product_state, run_network, psb_probs, leakage, init_thetas
    for enc, n in [('eo', 2), ('st', 3), ('eo', 3)]:
        N = (2 if enc == 'st' else 3) * n
        sched = bubble_schedule(N, N + 3)
        th = init_thetas(sched, 'random')
        for which in (['0', 'neel', 'alt'] if enc == 'st' else ['0']):
            psi_d = run_network(product_state(enc, n, which), th, sched, N)
            S = SectorSim(enc, n)
            psi_s = S.run(S.product_state(which), th, sched)
            dense_in_sector = psi_d.reshape(-1)[torch.tensor(S.basis)]
            e1 = (dense_in_sector - psi_s).abs().max().item()
            e2 = (psb_probs(psi_d, enc, n) - S.psb_probs(psi_s)).abs().max().item()
            e3 = abs(float(leakage(psi_d, enc, n)) - S.leakage(psi_s))
            print(f"{enc} n={n} init {which}: D={S.D}  |dpsi|={e1:.1e}  |dp|={e2:.1e}  |dleak|={e3:.1e}")
