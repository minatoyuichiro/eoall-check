"""Figures for the paper (paper/src/fig_*.pdf)."""
import json, glob, re, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = '../paper/src/'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7})

# ---- Fig 1: schedule schematic + pipeline -------------------------------------------
def fig_schedule():
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]; N, R = 6, 6
    for i in range(N):
        ax.plot([-.5, R - .5], [i, i], color='0.6', lw=0.8, zorder=1)
    for r in range(R):
        for i in range(r % 2, N - 1, 2):
            ax.plot([r, r], [i, i + 1], color='C0', lw=3, solid_capstyle='round', zorder=2)
            ax.text(r, i + .5, r'$\theta$', color='w', ha='center', va='center', fontsize=5, zorder=3)
    ax.set_xticks(range(R)); ax.set_xticklabels([f'{r+1}' for r in range(R)]); ax.set_xlabel('round $r$')
    ax.set_yticks(range(N)); ax.set_yticklabels([f'dot {i}' for i in range(N)]); ax.set_ylim(-.6, N - .4)
    for sp in ['top', 'right', 'left']: ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(r'bubble-sort schedule, $N=6$, $R=N$', fontsize=7)
    ax = axes[1]; ax.axis('off')
    # teacher at the top; two independent branches: MPS (classical control) and the exchange network -> PSB
    bx = dict(boxstyle='round,pad=0.3', ec='0.4', lw=0.6)
    ax.text(0.5, 0.92, 'GPT-2 teacher $p_t$\n(restricted vocab.)', ha='center', va='center', fontsize=6.3, bbox=dict(fc='#eef3fb', **bx), transform=ax.transAxes)
    ax.text(0.22, 0.50, 'MPS Born\nmachine ($\\chi$)\n(classical control)', ha='center', va='center', fontsize=6.0, bbox=dict(fc='#eef3fb', **bx), transform=ax.transAxes)
    ax.text(0.74, 0.55, 'exchange network\n(bubble-sort, $R$ rounds)', ha='center', va='center', fontsize=6.3, bbox=dict(fc='#fbeeee', **bx), transform=ax.transAxes)
    ax.text(0.74, 0.12, 'PSB readout $p(b)$', ha='center', va='center', fontsize=6.3, bbox=dict(fc='#fbeeee', **bx), transform=ax.transAxes)
    ax.annotate('', xy=(0.26, 0.66), xytext=(0.42, 0.84), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.annotate('', xy=(0.70, 0.68), xytext=(0.58, 0.84), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.annotate('', xy=(0.74, 0.22), xytext=(0.74, 0.44), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.text(0.26, 0.78, 'KL', fontsize=6, transform=ax.transAxes); ax.text(0.66, 0.78, 'KL', fontsize=6, transform=ax.transAxes)
    ax.text(0.5, 0.0, 'KL$(p_t\\|p)$ trained end-to-end', ha='center', fontsize=5.5, transform=ax.transAxes)
    fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_schedule.pdf'); fig.savefig(OUT + 'fig_schedule.png', dpi=160); plt.close(fig)

# ---- Fig 2: reach vs rounds (state prep n=2,3; unitary n=2) ---------------------------
def reach_points(fn):
    if os.path.exists(fn):
        d = [r for r in json.load(open(fn)) if r['sched'].startswith('bubble')]
        return [(int(r['sched'].split('=')[1]), r['F_haar']) for r in d]
    log = fn.replace('.json', '.log'); pts = []
    if os.path.exists(log):
        for ln in open(log):
            m = re.search(r'bubble R=(\d+)\s+p=\s*\d+\s+F_haar=([\d.]+)', ln)
            if m: pts.append((int(m.group(1)), float(m.group(2))))
    return pts

FLOOR = 1e-7
def unitary_points(n):
    """worst-of-4 process fidelity per R from results/e4_eo_n{n}.json, or from the log while the run is in progress."""
    fn = f'results/e4_eo_n{n}.json'
    if os.path.exists(fn):
        d = json.load(open(fn))
    else:
        d = [{'R': int(m.group(1)), 'F_proc': float(m.group(2))} for m in
             re.finditer(r'bubble R=(\d+)\s+p=\s*\d+\s+\S+\s*: F_proc=([\d.]+)', open(f'results/e4_eo_n{n}.log').read())] \
            if os.path.exists(f'results/e4_eo_n{n}.log') else []
    Rs = sorted(set(r['R'] for r in d))
    Rs = [R for R in Rs if sum(1 for r in d if r['R'] == R) == 4]  # only complete rounds
    return Rs, [min(r['F_proc'] for r in d if r['R'] == R) for R in Rs]

def fig_reach():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for fn, lab, mk in [('results/e2c_eo_n2.json', 'state prep., $n=2$ ($N=6$)', 'o'), ('results/e2c_eo_n3.json', 'state prep., $n=3$ ($N=9$)', 's')]:
        pts = reach_points(fn)
        if not pts: continue
        R, F = zip(*sorted(pts))
        ax.semilogy(R, np.maximum(1 - np.array(F), FLOOR), marker=mk, ms=4, label=lab)
    for n, N, mk in [(2, 6, '^'), (3, 9, 'v')]:
        Rs, F = unitary_points(n)
        if Rs:
            ax.semilogy(Rs, np.maximum(1 - np.array(F), FLOOR), marker=mk, ms=4, label=f'unitary, $n={n}$ ($N={N}$), worst of 4')
    ax.set_xlabel('rounds $R$'); ax.set_ylabel('$1-F$  (floor $10^{-7}$)'); ax.set_ylim(FLOOR * 0.5, 1)
    for x in (6, 9, 24): ax.axvline(x, color='0.7', lw=0.6, ls=':')
    ax.text(6.2, 3e-7, '$R{=}N$', fontsize=6, color='0.4'); ax.text(24.2, 3e-7, '$R{=}4N$', fontsize=6, color='0.4')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, columnspacing=0.8, handlelength=1.5); ax.grid(alpha=0.3, which='major')
    fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_reach.pdf'); fig.savefig(OUT + 'fig_reach.png', dpi=160); plt.close(fig)

# ---- Fig 3: distillation KL by configuration -------------------------------------------
def best(prefix, default_steps_only=False):
    """min KL over pulse initialisations (random/pi/zero/mps4) [and step counts] for a given (enc, teacher, init, schedule)."""
    files = [f for f in glob.glob(f'results/e3c_{prefix}_*.json') if not (default_steps_only and '_s' in f.split(prefix)[1])]
    vals = [json.load(open(f))['kl_final'] for f in files]
    return min(vals) if vals else None

def fig_distil():
    mps = json.load(open('results/e3_mps.json'))
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), sharey=True)
    for ax, T in zip(axes, ['cat_V8L2', 'begin_V4L3']):
        cfgs = [('st_%s_1_bubble24' % T, 'S-T $|T_0\\rangle^{\\otimes 6}$, $R{=}24$', 'C3'),
                ('st_%s_alt_bubble12' % T, 'S-T alt, $R{=}12$', 'C1'),
                ('st_%s_alt_bubble24' % T, 'S-T alt, $R{=}24$', 'C1'),
                ('st_%s_neel_bubble12' % T, 'S-T N\u00e9el, $R{=}12$', 'C2'),
                ('st_%s_neel_bubble24' % T, 'S-T N\u00e9el, $R{=}24$', 'C2'),
                ('eo_%s_0_bubble18' % T, 'EO $|0_L\\rangle^{\\otimes 6}$, $R{=}18$', 'C0'),
                ('eo_%s_0_bubble36' % T, 'EO $|0_L\\rangle^{\\otimes 6}$, $R{=}36$', 'C0')]
        xs, ys, cs, labs = [], [], [], []
        for tag, lab, c in cfgs:
            v = best(tag)
            if v is None: continue
            xs.append(len(xs)); ys.append(v); cs.append(c); labs.append(lab)
        ax.bar(xs, ys, color=cs, width=0.7)
        for chi in [1, 2, 4]:
            v = max(mps[T][f'chi{chi}']['kl'], 5e-4)
            ax.axhline(v, color='0.3', lw=0.6, ls='--'); ax.text(len(xs) - 0.5, v * 1.2, 'MPS $\\chi{=}%d$' % chi, fontsize=5.5, ha='right')
        ax.set_yscale('log'); ax.set_ylim(5e-4, 10)
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=70, ha='right', fontsize=5.5)
        ax.set_title({'cat_V8L2': '"The cat sat on", $V{=}8$, $L{=}2$', 'begin_V4L3': '"In the beginning", $V{=}4$, $L{=}3$'}[T], fontsize=6)
    axes[0].set_ylabel(r'KL$(p_t\,\|\,p_\theta)$ [nat]')
    fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_distil.pdf'); fig.savefig(OUT + 'fig_distil.png', dpi=160); plt.close(fig)


# ---- Fig 4: KL vs depth ---------------------------------------------------------------
def fig_depth():
    mps = json.load(open('results/e3_mps.json'))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    series = [('st_cat_V8L2_neel_bubble%d', 'S-T N\u00e9el, cat', 'C2', 'o'),
              ('st_cat_V8L2_alt_bubble%d', 'S-T alt, cat', 'C1', 's'),
              ('eo_cat_V8L2_0_bubble%d', 'EO, cat', 'C0', '^'),
              ('st_begin_V4L3_neel_bubble%d', 'S-T N\u00e9el, begin', 'C2', 'o'),
              ('st_begin_V4L3_alt_bubble%d', 'S-T alt, begin', 'C1', 's'),
              ('eo_begin_V4L3_0_bubble%d', 'EO, begin', 'C0', '^')]
    for pat, lab, c, mk in series:
        pts = [(R, best(pat % R, default_steps_only=True)) for R in (6, 9, 12, 18, 24, 36)]
        pts = [(R, v) for R, v in pts if v is not None]
        if len(pts) < 2: continue
        R, v = zip(*pts)
        ax.semilogy(R, v, marker=mk, ms=4, color=c, ls='-' if 'cat' in pat else '--', label=lab)
    for T, ls in [('cat_V8L2', '-'), ('begin_V4L3', '--')]:
        v = max(mps[T]['chi4']['kl'], 5e-4)
        ax.axhline(v, color='0.3', lw=0.6, ls=ls)
    ax.text(36.5, max(mps['cat_V8L2']['chi4']['kl'], 5e-4) * 1.2, 'MPS $\\chi{=}4$', fontsize=5.5, ha='right')
    ax.set_xlabel('rounds $R$'); ax.set_ylabel('KL [nat]'); ax.set_xticks([6, 12, 18, 24, 36])
    ax.legend(ncol=2, fontsize=5.5); ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3); fig.savefig(OUT + 'fig_depth.pdf'); fig.savefig(OUT + 'fig_depth.png', dpi=160); plt.close(fig)

# ---- Fig 5: scaling n=6 -> n=8, KL vs R/N ---------------------------------------------
def fig_scale():
    mps = json.load(open('results/e3_mps.json'))
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0), sharey=True)
    cfg = [('cat', 'cat_V8L2', 'cat_V16L2'), ('begin', 'begin_V4L3', 'begin_V4L4')]
    for ax, (name, t6, t8) in zip(axs, cfg):
        for enc, n, T, N, c, mk, ls in [('st', 6, t6, 12, 'C2', 'o', '-'), ('st', 8, t8, 16, 'C2', 'o', '--'),
                                        ('eo', 6, t6, 18, 'C0', '^', '-'), ('eo', 8, t8, 24, 'C0', '^', '--')]:
            init = 'neel' if enc == 'st' else '0'
            pts = []
            for R in (6, 9, 12, 16, 18, 24, 32, 36, 48, 64, 96):
                v = best(f'{enc}_{T}_{init}_bubble{R}', default_steps_only=True)
                if v is not None: pts.append((R / N, v))
            if len(pts) < 1: continue
            x, v = zip(*pts)
            ax.semilogy(x, v, marker=mk, ms=3.5, color=c, ls=ls, lw=1, label=('S-T N\u00e9el' if enc == 'st' else 'EO') + f', $n={n}$')
        for T, ls in [(t6, '-'), (t8, '--')]:
            ax.axhline(max(mps[T]['chi4']['kl'], 5e-4), color='0.3', lw=0.6, ls=ls)
        ax.set_title(f'teacher {name}', fontsize=7); ax.set_xlabel('$R/N$'); ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 6])
    axs[0].set_ylabel('KL [nat]'); axs[0].legend(fontsize=5, loc='upper right')
    axs[1].text(6, max(mps['begin_V4L4']['chi4']['kl'], 5e-4) * 1.25, 'MPS $\\chi{=}4$', fontsize=5, ha='right')
    fig.tight_layout(pad=0.3, w_pad=0.5); fig.savefig(OUT + 'fig_scale.pdf'); fig.savefig(OUT + 'fig_scale.png', dpi=160); plt.close(fig)

if __name__ == '__main__':
    fig_schedule(); fig_reach(); fig_distil(); fig_depth(); fig_scale(); print('figs written')
