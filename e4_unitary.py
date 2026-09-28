"""E4: compile a target LOGICAL UNITARY (polar factor Q of a real weight block W = QP)
into a leakage-free 3-spin-EO bubble-sort exchange network.  Process fidelity vs rounds.
Targets: polar factors of random Gaussian real matrices, plus of a GPT-2 c_proj sub-block."""
import json, sys, time, math
import numpy as np, torch
from scipy.linalg import polar
from eo_net import *
torch.set_num_threads(4)

def logical_block(th, sched, N, B, n):
    cols = []
    for x in range(2 ** n):
        psi0 = B[:, x].reshape((2,) * N)
        cols.append(run_network(psi0, th, sched, N).reshape(-1))
    U = torch.stack(cols, dim=1)            # 2^N x 2^n
    return B.conj().T @ U                    # 2^n x 2^n logical block M

def compile_unitary(T, sched, N, B, n, seeds=3, iters=500):
    d = 2 ** n; Tt = torch.tensor(T, dtype=CDT)
    best = None
    for s in range(seeds):
        g = torch.Generator().manual_seed(200 + s)
        th = init_thetas(sched, 'random', gen=g)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, history_size=50, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10, tolerance_change=1e-14)
        def closure():
            opt.zero_grad()
            M = logical_block(th, sched, N, B, n)
            loss = 1.0 - (torch.trace(Tt.conj().T @ M).abs() ** 2) / d ** 2
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            M = logical_block(th, sched, N, B, n)
            F = float((torch.trace(Tt.conj().T @ M).abs() ** 2) / d ** 2)
            L = float(1.0 - (M.abs() ** 2).sum() / d)
        if best is None or F > best[0]:
            best = (F, L, th.detach().clone())
    return best

def gpt2_block(d):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained('gpt2')
    W = m.transformer.h[0].attn.c_proj.weight.detach().numpy()[:d, :d]
    return W

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    B, N = logical_basis('eo', n); d = 2 ** n
    rng = np.random.default_rng(0)
    targets = []
    for k in range(3):
        W = rng.standard_normal((d, d)); Q, P = polar(W); targets.append((f'rand{k}', Q, P))
    W = gpt2_block(d); Q, P = polar(W); targets.append(('gpt2_cproj', Q, P))
    for name, Q, P in targets:
        s = np.linalg.svd(W if name.startswith('gpt2') else Q, compute_uv=False)
        print(f"target {name}: det Q = {np.linalg.det(Q):.3f}, |P-I|_F/|P|_F = {np.linalg.norm(P-np.eye(d))/np.linalg.norm(P):.3f}")
    rounds = {2: [8, 12, 16, 20, 24, 28, 32], 3: [18, 27, 36, 45, 54]}[n]
    out = []
    for R in rounds:
        sched = bubble_schedule(N, R); t0 = time.time()
        for name, Q, P in targets:
            F, L, th = compile_unitary(Q, sched, N, B, n)
            out.append(dict(n=n, N=N, R=R, params=n_params(sched), target=name, F_proc=F, leak=L))
            print(f"eo n={n} N={N} bubble R={R:2d} p={n_params(sched):3d} {name:10s}: F_proc={F:.6f} leak={L:.1e} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(f'results/e4_eo_n{n}.json', 'w'), indent=1)
