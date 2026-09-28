"""E5: pulse-angle noise robustness of trained Born machines.
Each exchange angle theta_k -> theta_k + delta_k, delta_k ~ N(0, sigma^2) (quasi-static charge noise
on J*t).  Report mean KL / TV over samples.  Uses thetas saved by e3_eo.py."""
import json, sys, glob, math
import numpy as np, torch
from eo_net import *
torch.set_num_threads(4)

def evaluate(rec, sigmas=(0.0, 0.01, 0.03, 0.1, 0.3), samples=20, seed=0):
    enc, n, N = rec['enc'], rec['n'], rec['N']
    T = json.load(open('results/e3_teachers.json'))[rec['teacher']]
    pt = torch.tensor(T['p'], dtype=RDT).reshape((2,) * n)
    kind, val = rec['sched'].split('=')
    sched = bubble_schedule(N, int(val)) if kind == 'bubble' else staircase_schedule(N, int(val))
    psi0 = product_state(enc, n, rec['init_state'])
    th0 = torch.tensor(rec['thetas'], dtype=RDT)
    g = torch.Generator().manual_seed(seed)
    out = {}
    with torch.no_grad():
        for s in sigmas:
            kls, tvs = [], []
            for _ in range(1 if s == 0 else samples):
                th = th0 + s * torch.randn(th0.shape, generator=g, dtype=RDT)
                p = psb_probs(run_network(psi0, th, sched, N), enc, n)
                kls.append(float(kl(pt, p))); tvs.append(float(0.5 * (p - pt).abs().sum()))
            out[s] = (float(np.mean(kls)), float(np.mean(tvs)))
    return out

if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('results/e3c_*.json'))
    res = {}
    for f in files:
        rec = json.load(open(f))
        if 'thetas' not in rec: continue
        out = evaluate(rec)
        tag = f.split('e3c_')[1][:-5]
        res[tag] = out
        print(tag + ": " + "  ".join(f"s={s}: KL {k:.4f} TV {t:.4f}" for s, (k, t) in out.items()), flush=True)
    json.dump({k: {str(s): v for s, v in o.items()} for k, o in res.items()}, open('results/e5_robust.json', 'w'), indent=1)
