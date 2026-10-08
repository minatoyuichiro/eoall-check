"""E3c: train exchange-only Born machines (2-spin S-T and 3-spin EO) on GPT-2 teacher
distributions.  Loss = KL(teacher || PSB-readout distribution).  Inits: random / zero / pi /
MPS-pretrained (state-level fidelity to the chi-4 MPS, then KL fine-tune)."""
import json, math, sys, time
import numpy as np, torch
from eo_net import *

def train_kl(psi0, th, sched, N, enc, n, pt, steps, lr=0.05):
    opt = torch.optim.Adam([th], lr=lr)
    best = (1e9, None); hist = []
    for it in range(steps):
        opt.zero_grad()
        psi = run_network(psi0, th, sched, N)
        p = psb_probs(psi, enc, n)
        loss = kl(pt, p)
        loss.backward()
        hist.append(loss.item())
        if loss.item() < best[0]:
            best = (loss.item(), th.detach().clone())   # parameters that produced this loss
        opt.step()
    return best, hist

def pretrain_fid(psi0, th, sched, N, target, iters=300):
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, history_size=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = 1.0 - fidelity(run_network(psi0, th, sched, N), target); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return float(fidelity(run_network(psi0, th, sched, N), target))

def main(enc, tname, init_state, sname, mode, steps, threads, seed=0):
    torch.set_num_threads(threads)
    T = json.load(open('results/e3_teachers.json'))[tname]
    M = json.load(open('results/e3_mps.json'))[tname]
    n = int(math.log2(len(T['p'])))
    b = 2 if enc == 'st' else 3; N = b * n
    pt = torch.tensor(T['p'], dtype=RDT).reshape((2,) * n)
    kind, val = sname.split('=')
    sched = bubble_schedule(N, int(val)) if kind == 'bubble' else staircase_schedule(N, int(val))
    psi0 = product_state(enc, n, init_state)
    g = torch.Generator().manual_seed(seed)
    rec = dict(enc=enc, teacher=tname, n=n, N=N, init_state=init_state, sched=sname, params=n_params(sched), init=mode, steps=steps, seed=seed)
    t0 = time.time()
    if mode.startswith('mps'):
        chi = mode[3:]
        c = torch.tensor(np.array(M[f'chi{chi}']['amp']), dtype=RDT).to(CDT)
        target = embed_logical(enc, n, c)
        th = init_thetas(sched, 'random', gen=g)
        F = pretrain_fid(psi0, th, sched, N, target)
        with torch.no_grad():
            kl0 = float(kl(pt, psb_probs(run_network(psi0, th, sched, N), enc, n)))
        rec.update(pretrain_fidelity=F, kl_after_pretrain=kl0, kl_teacher_mps=M[f'chi{chi}']['kl'])
        print(f"  pretrain: F(MPS chi={chi}) = {F:.4f}, KL after pretrain = {kl0:.4f} (teacher||MPS = {M[f'chi{chi}']['kl']:.4f})", flush=True)
        th = th.detach().clone().requires_grad_(True)
    else:
        th = init_thetas(sched, mode, gen=g)
    with torch.no_grad():
        rec['kl_init'] = float(kl(pt, psb_probs(run_network(psi0, th, sched, N), enc, n)))
    (best_kl, th_best), hist = train_kl(psi0, th, sched, N, enc, n, pt, steps)
    with torch.no_grad():
        psi = run_network(psi0, th_best, sched, N)
        p = psb_probs(psi, enc, n)
        rec['leak_final'] = float(leakage(psi, enc, n))
        rec['tv_final'] = float(0.5 * (p - pt).abs().sum())
    rec['kl_final'] = best_kl
    rec['hist'] = hist[::10]
    rec['thetas'] = th_best.tolist()
    rec['time_s'] = time.time() - t0
    print(f"{enc} {tname} init|{init_state}> {sname} p={rec['params']} {mode}: KL init {rec['kl_init']:.4f} -> final {best_kl:.4f}  "
          f"TV={rec['tv_final']:.4f} leak={rec['leak_final']:.3f} [{rec['time_s']:.0f}s]", flush=True)
    return rec

if __name__ == '__main__':
    enc, tname, init_state, sname, mode = sys.argv[1:6]
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    rec = main(enc, tname, init_state, sname, mode, steps, threads, seed)
    tag = f"{enc}_{tname}_{init_state}_{sname.replace('=','')}_{mode}" + (f"_s{steps}" if steps not in (300, 400) else "") + (f"_seed{seed}" if seed else "")
    json.dump(rec, open(f'results/e3c_{tag}.json', 'w'))
