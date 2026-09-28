"""Collect E7 results: best KL over initialisations, per (enc, teacher, R, kind)."""
import json, glob, collections
rows = collections.defaultdict(dict)
for f in glob.glob('results/e7_*.json'):
    r = json.load(open(f))
    key = (r['enc'], r['teacher'], r['R']); k = r['kind']
    cur = rows[key].get(k)
    if cur is None or r['kl_final'] < cur[0]:
        rows[key][k] = (r['kl_final'], r['init'], r['params'], r['tv_final'], r['leak_final'])
kinds = ['bubble', 'grid', 'roundrobin', 'randmatch']
for key in sorted(rows):
    print(key, '  '.join(f"{k}: {rows[key][k][0]:.4f}({rows[key][k][1][0]},p={rows[key][k][2]})" if k in rows[key] else f"{k}: -" for k in kinds))
