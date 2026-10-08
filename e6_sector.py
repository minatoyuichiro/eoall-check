"""E6: distillation on the sector-restricted simulator (eo_sector.py) for chains beyond
state-vector reach (eo n=8, N=24).  Same objective / optimiser / inits as e3_eo.py.
usage: python3 e6_sector.py enc teacher init_state bubble=R init steps threads"""
import json, math, sys, time
import numpy as np, torch
from eo_net import bubble_schedule, staircase_schedule, n_params, init_thetas, kl, RDT
from eo_sector import SectorSim

def main(enc, tname, init_state, sname, mode, steps, threads):
    torch.set_num_threads(threads)
    T = json.load(open('results/e3_teachers.json'))[tname]
    n = int(math.log2(len(T['p'])))
    S = SectorSim(enc, n); N = S.N
    pt = torch.tensor(T['p'], dtype=RDT).reshape((2,) * n)
    kind, val = sname.split('=')
    sched = bubble_schedule(N, int(val)) if kind == 'bubble' else staircase_schedule(N, int(val))
    psi0 = S.product_state(init_state)
    th = init_thetas(sched, mode, gen=torch.Generator().manual_seed(0))
    rec = dict(enc=enc, teacher=tname, n=n, N=N, D=S.D, init_state=init_state, sched=sname, params=n_params(sched), init=mode, steps=steps, sim='sector')
    t0 = time.time()
    with torch.no_grad():
        rec['kl_init'] = float(kl(pt, S.psb_probs(S.run(psi0, th, sched))))
    opt = torch.optim.Adam([th], lr=0.05)
    best = (1e9, None); hist = []
    for it in range(steps):
        opt.zero_grad()
        loss = kl(pt, S.psb_probs(S.run(psi0, th, sched)))
        loss.backward()
        hist.append(loss.item())
        if loss.item() < best[0]: best = (loss.item(), th.detach().clone())   # parameters that produced this loss
        opt.step()
        if it % 50 == 0: print(f"  step {it}: KL {loss.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
    with torch.no_grad():
        psi = S.run(psi0, best[1], sched); p = S.psb_probs(psi)
        rec['leak_final'] = S.leakage(psi); rec['tv_final'] = float(0.5 * (p - pt).abs().sum())
    rec['kl_final'] = best[0]; rec['hist'] = hist[::10]; rec['thetas'] = best[1].tolist(); rec['time_s'] = time.time() - t0
    print(f"{enc} {tname} init|{init_state}> {sname} p={rec['params']} {mode}: KL init {rec['kl_init']:.4f} -> final {best[0]:.4f}  "
          f"TV={rec['tv_final']:.4f} leak={rec['leak_final']:.3f} [{rec['time_s']:.0f}s]", flush=True)
    return rec

if __name__ == '__main__':
    enc, tname, init_state, sname, mode = sys.argv[1:6]
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    rec = main(enc, tname, init_state, sname, mode, steps, threads)
    tag = f"{enc}_{tname}_{init_state}_{sname.replace('=','')}_{mode}" + (f"_s{steps}" if steps not in (300, 400) else "")
    json.dump(rec, open(f'results/e3c_{tag}.json', 'w'))
