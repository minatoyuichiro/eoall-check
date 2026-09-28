"""E2c: minimum bubble-sort depth for exact logical state preparation (3-spin EO)."""
import json, sys, time
import torch
from eo_net import *
from e2_reach import optimise
torch.set_num_threads(4)

def run(n, rounds_list, stair_layers):
    enc = 'eo'
    B, N = logical_basis(enc, n)
    gen = torch.Generator().manual_seed(7)
    targets = [('haar', haar_logical(n, gen)) for _ in range(6)] + [('mps2', random_mps_logical(n, 2, gen)) for _ in range(6)]
    targets = [(k, embed_logical(enc, n, c)) for k, c in targets]
    psi0 = product_state(enc, n, '0')
    out = []
    cfgs = [(f'bubble R={R}', bubble_schedule(N, R)) for R in rounds_list] + \
           [(f'stair L={L}', staircase_schedule(N, L)) for L in stair_layers]
    for sname, sched in cfgs:
        t0 = time.time(); Fs = {'haar': [], 'mps2': []}; Ls = []
        for kind, tgt in targets:
            F, th = optimise(psi0, tgt, sched, N, 'random', seeds=3, iters=400)
            with torch.no_grad():
                Ls.append(float(leakage(run_network(psi0, th, sched, N), enc, n)))
            Fs[kind].append(F)
        rec = dict(n=n, N=N, sched=sname, params=n_params(sched),
                   F_haar=sum(Fs['haar'])/6, Fmin_haar=min(Fs['haar']), F_mps2=sum(Fs['mps2'])/6, Fmin_mps2=min(Fs['mps2']),
                   leak=sum(Ls)/len(Ls))
        out.append(rec)
        print(f"eo n={n} N={N} {sname:14s} p={rec['params']:3d}  F_haar={rec['F_haar']:.5f}(min {rec['Fmin_haar']:.5f}) "
              f"F_mps2={rec['F_mps2']:.5f}(min {rec['Fmin_mps2']:.5f}) leak={rec['leak']:.1e} [{time.time()-t0:.0f}s]", flush=True)
    return out

if __name__ == '__main__':
    n = int(sys.argv[1])
    if n == 2:
        res = run(2, [6, 7, 8, 9, 10, 12], [3, 4, 5])
    else:
        res = run(3, [9, 14, 18, 22, 27, 32], [4, 9, 12])
    json.dump(res, open(f'results/e2c_eo_n{n}.json', 'w'), indent=1)
