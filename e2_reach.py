"""E2: reachability of logical target states by exchange networks.
For each encoding / initial product state / schedule / init mode, maximise
fidelity to random logical targets (Haar and chi=2 MPS); report best F and
final leakage.  Multi-restart LBFGS."""
import json, math, sys, time
import torch
from eo_net import *

torch.set_num_threads(8)

def optimise(psi0, target, sched, N, mode, seeds=3, iters=300):
    best = None
    for s in range(seeds):
        g = torch.Generator().manual_seed(100 + s)
        th = init_thetas(sched, mode, gen=g)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, history_size=50,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-14)
        def closure():
            opt.zero_grad()
            psi = run_network(psi0, th, sched, N)
            loss = 1.0 - fidelity(psi, target)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            psi = run_network(psi0, th, sched, N)
            F = float(fidelity(psi, target))
        if best is None or F > best[0]:
            best = (F, th.detach().clone())
    return best

def main(enc, n, ntargets=6):
    B, N = logical_basis(enc, n)
    gen = torch.Generator().manual_seed(7)
    targets = [('haar', haar_logical(n, gen)) for _ in range(ntargets)] + \
              [('mps2', random_mps_logical(n, 2, gen)) for _ in range(ntargets)]
    targets = [(k, embed_logical(enc, n, c)) for k, c in targets]
    inits = {'st': ['0', '1', 'alt'], 'eo': ['0']}[enc]
    scheds = {
        f'bubble R={N}': bubble_schedule(N, N),
        f'bubble R={2*N}': bubble_schedule(N, 2 * N),
        f'stair L={N//2}': staircase_schedule(N, N // 2),
    }
    out = []
    for w in inits:
        psi0 = product_state(enc, n, w)
        for sname, sched in scheds.items():
            for mode in ['zero', 'pi', 'random']:
                t0 = time.time()
                Fs, Ls = {'haar': [], 'mps2': []}, {'haar': [], 'mps2': []}
                for kind, tgt in targets:
                    F, th = optimise(psi0, tgt, sched, N, mode)
                    with torch.no_grad():
                        L = float(leakage(run_network(psi0, th, sched, N), enc, n))
                    Fs[kind].append(F); Ls[kind].append(L)
                rec = dict(enc=enc, n=n, N=N, init_state=w, sched=sname, params=n_params(sched), init=mode,
                           F_haar=sum(Fs['haar'])/len(Fs['haar']), Fmin_haar=min(Fs['haar']),
                           F_mps2=sum(Fs['mps2'])/len(Fs['mps2']), Fmin_mps2=min(Fs['mps2']),
                           leak_haar=sum(Ls['haar'])/len(Ls['haar']), leak_mps2=sum(Ls['mps2'])/len(Ls['mps2']))
                out.append(rec)
                print(f"{enc} n={n} N={N} init|{w}> {sname:14s} p={rec['params']:3d} {mode:6s} "
                      f"F_haar={rec['F_haar']:.4f}(min {rec['Fmin_haar']:.4f}) F_mps2={rec['F_mps2']:.4f}(min {rec['Fmin_mps2']:.4f}) "
                      f"leak={rec['leak_haar']:.2e}  [{time.time()-t0:.0f}s]", flush=True)
    return out

if __name__ == '__main__':
    enc, n = sys.argv[1], int(sys.argv[2])
    res = main(enc, n)
    json.dump(res, open(f'results/e2_{enc}_n{n}.json', 'w'), indent=1)
