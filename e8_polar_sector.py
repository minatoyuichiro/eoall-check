"""E8a（セクター版）: 論理ユニタリのコンパイルを S_z セクター（n=3: 84 次元）で行う高速版。
e4_unitary.logical_block と同じ量（論理ブロック M = B† 𝓑_R B、プロセス忠実度、リーク）を計算する。
"""
import json, sys, time
import numpy as np, torch
from scipy.linalg import polar
from scipy.stats import ortho_group
from eo_net import logical_basis, bubble_schedule, n_params, init_thetas, CDT, RDT
torch.set_num_threads(4)

def sector(B, N):
    nz = torch.nonzero(B.abs().sum(1) > 1e-12).flatten().tolist()
    pc = bin(nz[0]).count('1'); assert all(bin(x).count('1') == pc for x in nz)
    idx = [x for x in range(2 ** N) if bin(x).count('1') == pc]
    pos = {x: k for k, x in enumerate(idx)}
    Bs = B[idx, :]                                  # (D, 2^n)
    return idx, pos, Bs

def perms(sched, idx, pos, N):
    out = []
    for rnd in sched:
        for (i, j) in rnd:
            bi, bj = N - 1 - i, N - 1 - j
            p = []
            for x in idx:
                a, b = (x >> bi) & 1, (x >> bj) & 1
                y = x if a == b else x ^ (1 << bi) ^ (1 << bj)
                p.append(pos[y])
            out.append(torch.tensor(p))
    return out

def block(th, P, Bs):
    psi = Bs                                         # (D, d) 列 = 論理基底
    for k, p in enumerate(P):
        t = th[k]
        psi = torch.exp(1j * t / 4) * (torch.cos(t / 2) * psi - 1j * torch.sin(t / 2) * psi[p])
    return Bs.conj().T @ psi                         # (d, d)

def compile_Q(Q, sched, Bs, P, d, seeds=12, adam=200, lbfgs=400, tag=''):
    Qt = torch.tensor(Q, dtype=CDT); best = None; Fs = []
    for s in range(seeds):
        th = init_thetas(sched, 'random', gen=torch.Generator().manual_seed(700 + s)).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(adam):
            opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (torch.trace(Qt.conj().T @ M).abs() ** 2) / d ** 2; loss.backward(); opt.step()
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=lbfgs, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (torch.trace(Qt.conj().T @ M).abs() ** 2) / d ** 2; loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            M = block(th, P, Bs); F = float((torch.trace(Qt.conj().T @ M).abs() ** 2) / d ** 2); L = float(1 - (M.abs() ** 2).sum() / d)
        Fs.append(F)
        if best is None or F > best[0]: best = (F, L, M.numpy())
    return best, Fs

if __name__ == '__main__':
    n = int(sys.argv[1]); B, N = logical_basis('eo', n); d = 2 ** n
    idx, pos, Bs = sector(B, N); print(f"n={n} N={N} sector dim {len(idx)}", flush=True)
    rng = np.random.default_rng(0); targets = []
    for k in range(3):
        W = rng.standard_normal((d, d)); Q, Pm = polar(W); targets.append((f'rand{k}', W, Q, Pm))
    from e4_unitary import gpt2_block
    W = gpt2_block(d); Q, Pm = polar(W); targets.append(('gpt2_cproj', W, Q, Pm))
    if n == 3:
        Qs = np.kron(ortho_group.rvs(4, random_state=1), ortho_group.rvs(2, random_state=2)); targets.append(('prod_2x1', Qs, Qs, np.eye(d)))
        Qs = np.kron(np.kron(ortho_group.rvs(2, random_state=3), ortho_group.rvs(2, random_state=4)), ortho_group.rvs(2, random_state=5)); targets.append(('prod_1x1x1', Qs, Qs, np.eye(d)))
        targets.append(('identity', np.eye(d), np.eye(d), np.eye(d)))
    Rs = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else {2: [8, 16, 24], 3: [27, 36, 45, 54, 72, 90]}[n]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if len(sys.argv) > 4: targets = [t for t in targets if t[0] in sys.argv[4].split(',')]
    tag = '_'.join(str(r) for r in Rs)
    out = []
    for R in Rs:
        sched = bubble_schedule(N, R); P = perms(sched, idx, pos, N)
        for name, W, Q, Pm in targets:
            t0 = time.time(); (F, L, M), Fs = compile_Q(Q, sched, Bs, P, d, seeds=seeds)
            ph = np.trace(Q.conj().T @ M); M = M * (abs(ph) / ph)   # 大域位相を合わせる
            err = np.linalg.norm(M @ Pm - W) / np.linalg.norm(W)
            Pfit, *_ = np.linalg.lstsq(M, W, rcond=None); err_fit = np.linalg.norm(M @ Pfit - W) / np.linalg.norm(W)
            rec = dict(n=n, R=R, params=n_params(sched), target=name, F_proc=F, leak=L, F_all=Fs, err_QP=float(err), err_QPfit=float(err_fit))
            out.append(rec)
            print(f"n={n} R={R:2d} p={n_params(sched):3d} {name:10s} F={F:.6f} leak={L:.1e} F_seeds[min/med]={min(Fs):.3f}/{np.median(Fs):.3f} err_QP={err:.3f} err_fit={err_fit:.3f} [{time.time()-t0:.0f}s]", flush=True)
            json.dump(out, open(f'results/e8_polar_sector_n{n}_R{tag}.json', 'w'), indent=1)
