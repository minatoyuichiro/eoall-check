"""Figure: KL versus depth R/N for 6, 10 and 12 logical qubits (two-spin Neel and three-spin), with the
chi=3 and chi=4 MPS levels of each teacher.  Reads results/e3c_*.json, results/e6_*.json? (n=8 not used),
results/e10_*.json and results/e3_mps.json.  Writes paper/src/fig_e10.pdf."""
import json, glob, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = json.load(open('results/e3_mps.json'))
# (family, n) -> teacher name
fam = {('cat', 6): 'cat_V8L2', ('cat', 10): 'cat_V32L2', ('cat', 12): 'cat_V64L2',
       ('begin', 6): 'begin_V4L3', ('begin', 10): 'begin_V4L5', ('begin', 12): 'begin_V8L4'}
runs = collections.defaultdict(list)   # (teacher, enc, init_state, R/N) -> [kl]
for f in glob.glob('results/e3c_*.json') + glob.glob('results/e10_*.json'):
    d = json.load(open(f))
    if d.get('steps') not in (300, 400): continue
    if not d['sched'].startswith('bubble'): continue
    if d['enc'] == 'st' and d['init_state'] != 'neel': continue
    if d['enc'] == 'eo' and d['init_state'] != '0': continue
    if d['init'].startswith('mps'): continue
    R = int(d['sched'].split('=')[1]); N = d['N']
    runs[(d['teacher'], d['enc'], R / N)].append(d.get('kl_saved', d['kl_final']))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
cols = {6: 'C0', 10: 'C1', 12: 'C3'}
for ax, family in zip(axes, ['cat', 'begin']):
    for n in (6, 10, 12):
        t = fam[(family, n)]
        for enc, mk, ls in (('st', 'o', '-'), ('eo', 's', '--')):
            xs = sorted({k[2] for k in runs if k[0] == t and k[1] == enc})
            if not xs: continue
            ys = [min(runs[(t, enc, x)]) for x in xs]        # best over initialisations / seeds
            ax.plot(xs, ys, marker=mk, ls=ls, color=cols[n], label=f'{"S–T Néel" if enc == "st" else "EO"}, $n={n}$')
        for chi, lsty in ((3, ':'), (4, '-.')):
            ax.axhline(M[t][f'chi{chi}']['kl'], color=cols[n], ls=lsty, lw=0.8, alpha=0.7)
    ax.set_yscale('log'); ax.set_xlabel('rounds $R/N$'); ax.set_title({'cat': '"The cat sat on"', 'begin': '"In the beginning"'}[family], fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('KL (nat)')
h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
for hh, ll in zip(h2, l2):
    if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, fontsize=7, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.03), frameon=False)
fig.text(0.5, -0.09, 'horizontal lines: MPS at $\\chi=3$ (dotted) and $\\chi=4$ (dash-dotted) for each teacher', ha='center', fontsize=7)
fig.tight_layout()
fig.savefig('../paper/src/fig_e10.pdf', bbox_inches='tight'); fig.savefig('../paper/src/fig_e10.png', dpi=150, bbox_inches='tight')
for k in sorted(runs): print(k, [round(v, 3) for v in sorted(runs[k])])
