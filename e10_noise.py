"""Pulse-area noise at 10-12 logical qubits (as Table V): add N(0, sigma^2) to every stored pulse area of an E10
run and re-evaluate KL/TV; mean over samples.  usage: python3 e10_noise.py results/e10_<tag>.json samples threads"""
import json, sys, math
import numpy as np, torch
from eo_net import bubble_schedule, RDT, CDT
from e10_sector import BigSectorSim
f = sys.argv[1]; samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20; torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
d = json.load(open(f)); T = json.load(open('results/e3_teachers.json'))[d['teacher']]
n = d['n']; S = BigSectorSim(d['enc'], n); N = S.N
pt = torch.tensor(T['p'], dtype=RDT); pt = pt / pt.sum()
sched = bubble_schedule(N, int(d['sched'].split('=')[1])); psi0 = S.product_state(d['init_state'])
th0 = torch.tensor(d['thetas'], dtype=RDT)
def evalkl(th):
    p = S.psb_probs(S.run(psi0, th, sched)); return float((pt * (torch.log(pt + 1e-12) - torch.log(p + 1e-12))).sum()), float(0.5 * (p - pt).abs().sum())
out = dict(file=f, kl0=evalkl(th0)[0], samples=samples, sigma={})
print(f"{f}: noiseless KL {out['kl0']:.4f} (stored {d['kl_final']:.4f})", flush=True)
g = torch.Generator().manual_seed(0)
for sig in (0.01, 0.03, 0.1):
    kls, tvs = [], []
    for s in range(samples):
        kl, tv = evalkl(th0 + sig * torch.randn(len(th0), generator=g, dtype=RDT)); kls.append(kl); tvs.append(tv)
    out['sigma'][str(sig)] = dict(kl_mean=float(np.mean(kls)), kl_std=float(np.std(kls)), tv_mean=float(np.mean(tvs)), kl=kls)
    print(f"  sigma {sig}: KL {np.mean(kls):.4f} +- {np.std(kls):.4f}  TV {np.mean(tvs):.4f}", flush=True)
json.dump(out, open(f.replace('results/e10_', 'results/e10noise_'), 'w'))
