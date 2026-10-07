"""Table IV with seed statistics (review 2026-10-01, M4).
Per (enc, teacher, R, layout): for each seed take the better of the available initialisations (pi / random),
then report median [min, max] over seeds and the number of seeds."""
import json, glob, collections, numpy as np
runs = collections.defaultdict(lambda: collections.defaultdict(dict))   # key -> kind -> seed -> best kl
for f in glob.glob('results/e7_*.json'):
    if 'table' in f: continue
    r = json.load(open(f))
    key = (r['enc'], r['teacher'], r['R']); k = r['kind']; s = r.get('seed', 0)
    cur = runs[key][k].get(s)
    if cur is None or r['kl_final'] < cur:
        runs[key][k][s] = r['kl_final']
kinds = ['bubble', 'grid', 'roundrobin', 'randmatch']
out = {}
for key in sorted(runs):
    cells = []
    for k in kinds:
        v = np.array(sorted(runs[key][k].values()))
        if len(v) == 0: cells.append('-'); continue
        out[str(key) + k] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()), n=int(len(v)), best=float(v.min()))
        cells.append(f"{np.median(v):.3f} [{v.min():.3f}-{v.max():.3f}] n={len(v)}")
    print(key, ' | '.join(f"{k}: {c}" for k, c in zip(kinds, cells)))
json.dump(out, open('results/e7_table_seeds.json', 'w'), indent=1)
