"""Distribution-level SU(2) ceiling for 2-spin S-T distillation (e3c control).
Exchange dynamics keep the per-sector norms c_S = ||P_S psi0|| fixed and act as an
arbitrary unitary inside each (S, Sz) block.  So the reachable set of states is
    { sum_S c_S phi_S : phi_S arbitrary unit vector in sector (S, Sz=0) }.
Minimising KL(teacher || PSB(psi)) over that set (no circuit at all) gives the best
any exchange-only network can do from psi0.  Compare with e3c bubble results."""
import json, sys, time, math
import numpy as np, torch
from eo_net import *
from e2b_ceiling import spin_ops

def sector_bases(N, M0):
    import os
    cache = f'results/sectors_N{N}_M{M0}.pt'
    if os.path.exists(cache): return torch.load(cache, weights_only=False)
    S2, Sz = spin_ops(N)
    w, V = np.linalg.eigh(S2 + 1e-3 * Sz)
    S2w = np.real(np.einsum('ij,ij->j', V.conj(), S2 @ V))
    Svals = np.round((-1 + np.sqrt(1 + 4 * S2w)) / 2, 6)
    Mvals = np.round(np.real(np.einsum('ij,ij->j', V.conj(), Sz @ V)), 6)
    out = {}
    for S in sorted(set(Svals)):
        idx = np.where((Svals == S) & (np.abs(Mvals - M0) < 1e-6))[0]
        if len(idx): out[S] = torch.tensor(V[:, idx], dtype=CDT)
    torch.save(out, cache)
    return out

def main(tname, init, steps=1500, seeds=3, threads=4):
    torch.set_num_threads(threads)
    enc, n = 'st', 6; N = 2 * n
    teach = json.load(open('results/e3_teachers.json'))[tname]
    pt = torch.tensor(np.array(teach['p'], dtype=np.float64)).reshape((2,) * n)
    bases = sector_bases(N, 0.0)
    free = init == 'free'                # optimise the sector weights too (= best over all initial states in Sz=0)
    if free:
        cS = {S: 1.0 / math.sqrt(len(bases)) for S in bases}
    else:
        psi0 = product_state(enc, n, init).reshape(-1)
        cS = {S: (Vs.conj().T @ psi0).norm().item() for S, Vs in bases.items()}
    print(f"{tname} init|{init}>: sector weights c_S^2 = " + ", ".join(f"S={S:.0f}:{c*c:.3f}(m={Vs.shape[1]})" for (S, Vs), c in zip(bases.items(), cS.values())), flush=True)
    best = None
    for seed in range(seeds):
        gen = torch.Generator().manual_seed(seed)
        params = {S: torch.nn.Parameter(torch.randn(Vs.shape[1], 2, generator=gen, dtype=RDT)) for S, Vs in bases.items() if cS[S] > 1e-6}
        plist = list(params.values())
        if free:
            wlog = torch.nn.Parameter(torch.zeros(len(bases), dtype=RDT)); plist.append(wlog)
        opt = torch.optim.Adam(plist, lr=0.02)
        t0 = time.time()
        for it in range(steps):
            psi = 0
            if free:
                w = torch.softmax(wlog, 0).sqrt()
            for k, (S, p) in enumerate(params.items()):
                v = torch.complex(p[:, 0], p[:, 1]); v = v / v.norm()
                psi = psi + (w[k] if free else cS[S]) * (bases[S] @ v)
            pm = psb_probs(psi.reshape((2,) * N), enc, n)
            loss = kl(pt, pm)
            opt.zero_grad(); loss.backward(); opt.step()
        tv = 0.5 * (pt - pm).abs().sum().item()
        extra = ("  w=" + ",".join(f"{x:.3f}" for x in torch.softmax(wlog, 0).tolist())) if free else ""
        print(f"  seed {seed}: KL ceiling = {loss.item():.4f}  TV={tv:.4f} [{time.time()-t0:.0f}s]{extra}", flush=True)
        if best is None or loss.item() < best[0]: best = (loss.item(), tv)
    return best

if __name__ == '__main__':
    res = {}
    inits = sys.argv[1].split(',') if len(sys.argv) > 1 else ['alt', '1', '0']
    out = sys.argv[2] if len(sys.argv) > 2 else 'results/e3d_dist_ceiling.json'
    for tname in ['cat_V8L2', 'begin_V4L3']:
        for init in inits:
            res[f"{tname}_{init}"] = main(tname, init, seeds=2)
    json.dump(res, open(out, 'w'), indent=1)
