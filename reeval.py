"""Re-evaluate KL/TV/leakage from the SAVED parameters of every stored run (review 2026-10-07, Codex finding 1:
the stored parameters were one optimiser step past the recorded minimum).  Adds kl_saved/tv_saved/leak_saved
to each JSON and prints the largest discrepancies."""
import json, glob, math, sys
import numpy as np, torch
from eo_net import *
from eo_sector import SectorSim
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
pattern = sys.argv[1] if len(sys.argv) > 1 else 'results/e3c_*.json'
T = json.load(open('results/e3_teachers.json'))
worst = []
for f in sorted(glob.glob(pattern)):
    d = json.load(open(f))
    if 'thetas' not in d or 'kl_saved' in d: continue
    enc, n, t = d['enc'], d['n'], d['teacher']; N = d['N']
    pt = torch.tensor(T[t]['p'], dtype=RDT)
    th = torch.tensor(d['thetas'], dtype=RDT)
    if 'sched' in d:
        kind, val = d['sched'].split('=') if '=' in d['sched'] else (d['sched'], None)
    else:                                           # e7 records: 'kind' and 'R'
        kind, val = 'bubble', str(d['R'])
    if d.get('kind', 'bubble') != 'bubble':         # e7 non-chain layouts
        from e7_connect import make_schedule, run_general
        sched = make_schedule(d['kind'], N, 2 if enc == 'st' else 3, d['R'])
        psi = run_general(product_state(enc, n, d['init_state']), th, sched, N); p = psb_probs(psi, enc, n).reshape(-1); leak = leakage(psi, enc, n)
    elif N <= 18:
        sched = bubble_schedule(N, int(val)) if kind == 'bubble' else staircase_schedule(N, int(val))
        psi = run_network(product_state(enc, n, d['init_state']), th, sched, N); p = psb_probs(psi, enc, n).reshape(-1); leak = leakage(psi, enc, n)
    else:
        S = SectorSim(enc, n); sched = bubble_schedule(N, int(val))
        psi = S.run(S.product_state(d['init_state']), th, sched); p = S.psb_probs(psi).reshape(-1); leak = S.leakage(psi)
    kl_s = float(kl(pt, p)); tv_s = float(0.5 * (p - pt).abs().sum())
    d['kl_saved'], d['tv_saved'], d['leak_saved'] = kl_s, tv_s, float(leak)
    json.dump(d, open(f, 'w'))
    worst.append((abs(kl_s - d['kl_final']), f.split('/')[-1], d['kl_final'], kl_s, d.get('tv_final'), tv_s))
worst.sort(reverse=True)
for w in worst[:12]: print(f"{w[1]:55s} kl_rec {w[2]:.5f} kl_saved {w[3]:.5f} diff {w[0]:.5f} tv {w[4]:.4f}->{w[5]:.4f}")
print('files', len(worst), 'max |diff|', worst[0][0] if worst else None)
