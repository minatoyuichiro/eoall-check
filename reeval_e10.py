"""Re-evaluate E10 records (BigSectorSim, CPU) from stored parameters: adds kl_saved/tv_saved."""
import json, glob, sys, torch
from eo_net import bubble_schedule, RDT
from e10_sector import BigSectorSim
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
T = json.load(open('results/e3_teachers.json')); sims = {}
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results/e10_*.json')):
    d = json.load(open(f))
    if 'thetas' not in d or 'kl_saved' in d: continue
    key = (d['enc'], d['n']); S = sims.get(key) or BigSectorSim(d['enc'], d['n']); sims[key] = S
    pt = torch.tensor(T[d['teacher']]['p'], dtype=RDT); pt = pt / pt.sum()
    th = torch.tensor(d['thetas'], dtype=RDT); sched = bubble_schedule(S.N, int(d['sched'].split('=')[1]))
    p = S.psb_probs(S.run(S.product_state(d['init_state']), th, sched))
    d['kl_saved'] = float((pt * (torch.log(pt + 1e-12) - torch.log(p + 1e-12))).sum()); d['tv_saved'] = float(0.5 * (p - pt).abs().sum())
    json.dump(d, open(f, 'w')); print(f"{f.split('/')[-1]:50s} rec {d['kl_final']:.5f} saved {d['kl_saved']:.5f} diff {d['kl_saved']-d['kl_final']:+.5f}", flush=True)
