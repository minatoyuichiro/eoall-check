"""E10: distillation at 10-14 logical qubits.

Large-N sector simulator (no 2^N arrays: basis enumerated combinatorially, lookups by
searchsorted) with a hand-written adjoint gradient (two state vectors in memory instead
of one per pulse), device-agnostic (CPU or CUDA).  Same conventions as eo_sector.py:
index bit 0 = up, site 0 = most significant bit, pulse on pair (i,i+1):
    psi' = e^{i a/4} [cos(a/2) psi - i sin(a/2) psi[perm]].
Readout: PSB on the first pair of every block (singlet -> 0, triplet -> 1).

usage: python3 e10_sector.py enc teacher init_state bubble=R init steps threads [seed] [device]
"""
import json, math, sys, time
import numpy as np, torch
from eo_net import bubble_schedule, n_params, init_thetas, RDT, CDT

def basis_fixed_weight(N, k):
    """sorted int64 array of all N-bit integers with exactly k bits set (vectorised recursion)."""
    tab = {(0, 0): np.array([0], dtype=np.int64)}
    def B(m, j):
        if j < 0 or j > m: return np.array([], dtype=np.int64)
        if (m, j) not in tab:
            tab[(m, j)] = np.concatenate([B(m - 1, j), B(m - 1, j - 1) | (1 << (m - 1))])
        return tab[(m, j)]
    return np.sort(B(N, k))

class BigSectorSim:
    def __init__(self, enc, n, device='cpu', lazy=False, cdt=CDT):
        self.enc, self.n, self.dev, self.lazy, self.cdt = enc, n, torch.device(device), lazy, cdt
        self.rdt = torch.float32 if cdt == torch.complex64 else RDT
        self.b = 2 if enc == 'st' else 3
        self.N = N = self.b * n
        self.basis = basis_fixed_weight(N, n)          # st: S_z=0 (n down of 2n); eo: S_z=n/2 (n down of 3n)
        self.D = len(self.basis)
        self.basis_t = torch.tensor(self.basis.astype(np.int32) if N <= 31 else self.basis, device=self.dev) if lazy else None
        self._perm = {}
        # readout bookkeeping (lazy mode: recomputed per call to save memory)
        self.ro = None if lazy else [self._pair(self.b * j) for j in range(n)]
        key = np.zeros(self.D, dtype=np.int64)
        for j in range(n):
            i = self.b * j
            bi = (self.basis >> (N - 1 - i)) & 1; bj = (self.basis >> (N - 2 - i)) & 1
            key += ((bi == 0) & (bj == 1) == False).astype(np.int64) << (n - 1 - j)
        self.key = torch.tensor(key, device=self.dev)

    def _pair(self, i):
        if i in self._perm: return self._perm[i]
        N = self.N
        mi, mj = 1 << (N - 1 - i), 1 << (N - 2 - i)
        if self.lazy:                                   # compute on device, do not cache (memory)
            b = self.basis_t
            diff = ((b & mi) != 0) != ((b & mj) != 0)
            y = torch.where(diff, b ^ (mi | mj), b)
            perm = torch.searchsorted(b, y, out_int32=(b.dtype == torch.int32)); del y
            sign = torch.where((b & mi) != 0, 1.0, -1.0).to(self.rdt)
            return (perm, diff, sign)
        bi = (self.basis & mi) != 0; bj = (self.basis & mj) != 0
        diff = bi != bj
        y = np.where(diff, self.basis ^ (mi | mj), self.basis)
        perm = torch.tensor(np.searchsorted(self.basis, y), device=self.dev)
        sign = torch.tensor(np.where(bi, 1.0, -1.0), dtype=self.rdt, device=self.dev)
        self._perm[i] = (perm, torch.tensor(diff, device=self.dev), sign)
        return self._perm[i]

    # ---------------------------------------------------------------- states
    def product_state(self, which):
        """product of per-block states; st: '0'=|S>, '1'=|T0>, 'alt', 'neel'=|up down> per pair; eo: '0'=|0_L>^n."""
        n, N, b = self.n, self.N, self.b
        amp = np.ones(self.D, dtype=np.complex128)
        for j in range(n):
            i = b * j
            bits = [((self.basis >> (N - 1 - i - t)) & 1) for t in range(b)]   # 0 = up, 1 = down
            if self.enc == 'st':
                kind = {'neel': 'neel', '0': 'S', '1': 'T', 'alt': ('S' if j % 2 == 0 else 'T')}.get(which)
                if kind is None: kind = 'S' if which[j] == '0' else 'T'
                b0, b1 = bits
                if kind == 'neel':  f = ((b0 == 0) & (b1 == 1)).astype(float)                         # |up down>
                elif kind == 'S':   f = np.where((b0 == 0) & (b1 == 1), 1, np.where((b0 == 1) & (b1 == 0), -1, 0)) / math.sqrt(2)
                else:               f = np.where((b0 == 0) & (b1 == 1), 1, np.where((b0 == 1) & (b1 == 0), 1, 0)) / math.sqrt(2)
            else:
                assert which == '0'
                b0, b1, b2 = bits                                                   # |S>_12 |up>_3
                f = np.where((b2 == 0) & (b0 == 0) & (b1 == 1), 1, np.where((b2 == 0) & (b0 == 1) & (b1 == 0), -1, 0)) / math.sqrt(2)
            amp *= f
        amp /= np.linalg.norm(amp)
        return torch.tensor(amp, dtype=self.cdt, device=self.dev)

    # ---------------------------------------------------------------- dynamics
    def _apply(self, psi, i, c0, c1):
        """c0 * psi + c1 * psi[perm] with two buffers (in-place where possible)"""
        perm = self._pair(i)[0]
        out = psi[perm]; out.mul_(c1); out.add_(psi, alpha=c0); return out
    def U(self, psi, i, a):
        a = float(a); ph = complex(math.cos(a / 4), math.sin(a / 4))
        return self._apply(psi, i, ph * math.cos(a / 2), ph * (-1j) * math.sin(a / 2))
    def Udag(self, psi, i, a):
        a = float(a); ph = complex(math.cos(a / 4), -math.sin(a / 4))
        return self._apply(psi, i, ph * math.cos(a / 2), ph * (1j) * math.sin(a / 2))
    def dU(self, psi, i, a):
        a = float(a); ph = complex(math.cos(a / 4), math.sin(a / 4))
        c0 = ph * ((1j / 4) * math.cos(a / 2) - 0.5 * math.sin(a / 2))
        c1 = ph * ((1j / 4) * (-1j) * math.sin(a / 2) - 0.5j * math.cos(a / 2))
        return self._apply(psi, i, c0, c1)

    def pulses(self, sched):
        return [i for rnd in sched for (i, j) in rnd]

    def run(self, psi0, thetas, sched):
        psi = psi0
        for k, i in enumerate(self.pulses(sched)):
            psi = self.U(psi, i, thetas[k])
        return psi

    # ---------------------------------------------------------------- readout
    def _readout_T(self, x, sgn=1.0):
        """the real orthogonal pair transform T (sgn=+1) or its transpose (sgn=-1), 2 buffers"""
        x = x.clone()
        for j in range(self.n):
            perm, diff, s = self.ro[j] if self.ro is not None else self._pair(self.b * j)
            g = x[perm]; g.mul_(sgn * s); g.add_(x); g.mul_(1 / math.sqrt(2))
            x = torch.where(diff, g, x); del g, perm, diff, s
        return x
    def psb_probs(self, psi):
        z = self._readout_T(psi)
        q = (z.abs() ** 2).to(RDT); del z
        return torch.zeros(2 ** self.n, dtype=RDT, device=self.dev).index_add(0, self.key, q)
    def psb_loss_grad(self, psi, pt, eps=1e-12):
        """L = KL_eps(pt||p(psi)) and lam = dL/dRe psi + i dL/dIm psi (torch convention), without autograd"""
        z = self._readout_T(psi)
        q = (z.abs() ** 2).to(RDT)
        p = torch.zeros(2 ** self.n, dtype=RDT, device=self.dev).index_add(0, self.key, q); del q
        L = float((pt * (torch.log(pt + eps) - torch.log(p + eps))).sum())
        w = (-2.0 * pt / (p + eps)).to(self.rdt)          # dL/dp_b times 2
        z.mul_(w[self.key]); lam = self._readout_T(z, sgn=-1.0); del z
        return L, lam

    # ---------------------------------------------------------------- loss and adjoint gradient
    def loss_and_grad(self, psi0, thetas, sched, pt, eps=1e-12):
        """KL_eps(pt || p(psi_R)) and dKL/dtheta by the adjoint method (unitary circuit: O(1) states)."""
        pl = self.pulses(sched); th = thetas.detach()
        with torch.no_grad():
            psi = psi0
            for k, i in enumerate(pl): psi = self.U(psi, i, th[k])
        L, lam = self.psb_loss_grad(psi, pt, eps)         # torch convention: dL/dtheta = Re <lam | dpsi/dtheta>
        g = torch.zeros(len(pl), dtype=RDT, device=self.dev)
        with torch.no_grad():
            for k in range(len(pl) - 1, -1, -1):
                i = pl[k]; a = th[k]
                psi_prev = self.Udag(psi, i, a); del psi
                d = self.dU(psi_prev, i, a); g[k] = torch.vdot(lam, d).real.to(RDT); del d
                lam = self.Udag(lam, i, a)
                psi = psi_prev
        return L, g

def main(enc, tname, init_state, sname, mode, steps, threads, seed=0, device='cpu', c64=False, tag=''):
    torch.set_num_threads(threads)
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(float(__import__('os').environ.get('EO_GPU_FRAC', '0.083')))
    T = json.load(open('results/e3_teachers.json'))[tname]
    n = int(math.log2(len(T['p'])))
    t0 = time.time()
    S = BigSectorSim(enc, n, device, lazy=device.startswith('cuda'), cdt=(torch.complex64 if c64 else CDT)); N = S.N
    print(f"sector built: enc={enc} n={n} N={N} D={S.D} [{time.time()-t0:.0f}s]", flush=True)
    pt = torch.tensor(T['p'], dtype=RDT, device=device); pt = pt / pt.sum()
    kind, val = sname.split('='); assert kind == 'bubble'
    sched = bubble_schedule(N, int(val))
    psi0 = S.product_state(init_state)
    th = init_thetas(sched, mode, gen=torch.Generator().manual_seed(seed)).detach().to(device).requires_grad_(True)
    rec = dict(enc=enc, teacher=tname, n=n, N=N, D=S.D, init_state=init_state, sched=sname, params=n_params(sched), init=mode, steps=steps, seed=seed, sim='bigsector_adjoint', device=device)
    opt = torch.optim.Adam([th], lr=0.05)
    best = (1e9, None); hist = []; start = 0
    ck = f'results/e10_ckpt_{tag}.pt'
    import os
    if tag and os.path.exists(ck):                    # resume (the GPU slot is interrupted daily)
        c = torch.load(ck, map_location=device)
        th.data.copy_(c['th']); opt.load_state_dict(c['opt']); best = (c['best'][0], c['best'][1].to(device)); hist = c['hist']; start = c['it'] + 1; rec['kl_init'] = c['kl_init']
        print(f"resumed from step {start}", flush=True)
    for it in range(start, steps):
        L, g = S.loss_and_grad(psi0, th, sched, pt)
        if it == 0: rec['kl_init'] = L
        hist.append(L)
        if L < best[0]: best = (L, th.detach().clone())   # parameters that produced this loss
        opt.zero_grad(); th.grad = g; opt.step()
        if it % 20 == 0: print(f"  step {it}: KL {L:.4f} [{time.time()-t0:.0f}s]", flush=True)
        if tag and it % 10 == 0:
            torch.save(dict(th=th.detach().cpu(), opt=opt.state_dict(), best=(best[0], best[1].cpu()), hist=hist, it=it, kl_init=rec['kl_init']), ck)
    with torch.no_grad():
        psi = S.run(psi0, best[1], sched); p = S.psb_probs(psi)
        rec['tv_final'] = float(0.5 * (p - pt).abs().sum())
    rec['kl_final'] = best[0]; rec['hist'] = hist[::10]; rec['thetas'] = best[1].cpu().tolist(); rec['time_s'] = time.time() - t0
    print(f"{enc} {tname} |{init_state}> {sname} {mode} seed{seed}: KL {rec['kl_init']:.4f} -> {best[0]:.4f}  TV={rec['tv_final']:.4f} [{rec['time_s']:.0f}s]", flush=True)
    return rec

if __name__ == '__main__':
    enc, tname, init_state, sname, mode = sys.argv[1:6]
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    device = sys.argv[9] if len(sys.argv) > 9 else 'cpu'
    c64 = len(sys.argv) > 10 and sys.argv[10] == 'c64'
    tag = f"{enc}_{tname}_{init_state}_{sname.replace('=','')}_{mode}" + (f"_s{steps}" if steps not in (300, 400) else "") + (f"_seed{seed}" if seed else "") + ("_c64" if c64 else "")
    rec = main(enc, tname, init_state, sname, mode, steps, threads, seed, device, c64, tag)
    json.dump(rec, open(f'results/e10_{tag}.json', 'w'))
