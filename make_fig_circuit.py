"""Explanatory figures (no data): the exchange pulse, the two encodings as circuit diagrams with PSB readout,
and the selection-rule picture (which total-spin sectors feed which Hamming classes of PSB outcomes).
Writes paper/src/fig_circuit.pdf (two-column) and paper/src/fig_sectors.pdf (one column)."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from math import comb
OUT = '../paper/src/'
plt.rcParams.update({'font.size': 7, 'mathtext.fontset': 'dejavusans'})

BLUE, RED, GREY = '#2a6fb0', '#b03a3a', '0.55'

def gate(ax, x, i, label=r'$\theta$', color=BLUE):
    """one exchange pulse between wires i and i+1 at time x"""
    ax.plot([x, x], [i, i + 1], color=color, lw=1.4, zorder=2)
    ax.add_patch(FancyBboxPatch((x - .31, i + .5 - .28), .62, .56, boxstyle='round,pad=0.02', fc=color, ec='none', zorder=3))
    ax.text(x, i + .5, label, color='w', ha='center', va='center', fontsize=6.5, zorder=4)

def psb(ax, x, i, label):
    """PSB readout box on the pair (i, i+1)"""
    ax.add_patch(FancyBboxPatch((x - .5, i - .3), 1.0, 1.6, boxstyle='round,pad=0.03', fc='#fbeeee', ec=RED, lw=0.8, zorder=3))
    ax.text(x, i + .5, 'PSB', ha='center', va='center', fontsize=6.5, color=RED, zorder=4)
    ax.annotate('', xy=(x + 1.1, i + .5), xytext=(x + .52, i + .5), arrowprops=dict(arrowstyle='->', lw=0.7, color=RED))
    ax.text(x + 1.15, i + .5, label, ha='left', va='center', fontsize=7.5, color=RED)

def panel(ax, N, R, block, init_labels, readout_labels, title, unread):
    for i in range(N):
        ax.plot([-.15, R + 1.3], [i, i], color=GREY, lw=0.8, zorder=1)
        ax.text(-.3, i, f'dot {i}', ha='right', va='center', fontsize=6.6)
    for r in range(R):
        for i in range(r % 2, N - 1, 2):
            gate(ax, r + .5, i, r'$\theta_{%d,%d}$' % (r, i) if N <= 6 and False else r'$\theta$')
    ax.text(R + .55, (N - 1) / 2, r'$\cdots$', ha='center', va='center', fontsize=9, color='0.3')
    ax.text(R / 2, N - .35, f'rounds $1\\ldots R$', ha='center', va='bottom', fontsize=6, color='0.3')
    # blocks: brace-like brackets and initial labels
    for j in range(N // block):
        lo, hi = j * block, j * block + block - 1
        ax.add_patch(Rectangle((-4.9, lo - .35), 3.1, hi - lo + .7, fc='#eef3fb', ec='0.5', lw=0.5, zorder=0))
        ax.text(-3.35, (lo + hi) / 2, init_labels[j], ha='center', va='center', fontsize=6.8, linespacing=1.4)
        psb(ax, R + 1.75, lo, readout_labels[j])
        if block == 3:
            ax.text(R + 2.25, hi, unread, ha='left', va='center', fontsize=6.2, color='0.4')
    ax.text(-3.35, N - .35, 'initial state', ha='center', va='bottom', fontsize=6, color='0.3')
    ax.text(R + 2.3, N - .35, 'readout', ha='center', va='bottom', fontsize=6, color='0.3')
    ax.set_xlim(-5.05, R + 3.9); ax.set_ylim(-1.2, N + .2); ax.axis('off')
    ax.set_title(title, fontsize=7.5, loc='left')

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[0.85, 1.25, 1.25]))
# (a) the pulse
ax = axes[0]; ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('(a) exchange pulse', fontsize=7.5, loc='left')
ax.plot([0.15, 0.85], [0.72, 0.72], color=GREY, lw=0.8); ax.plot([0.15, 0.85], [0.56, 0.56], color=GREY, lw=0.8)
ax.plot([0.5, 0.5], [0.56, 0.72], color=BLUE, lw=1.4)
ax.add_patch(FancyBboxPatch((0.45, 0.595), 0.10, 0.09, boxstyle='round,pad=0.01', fc=BLUE, ec='none'))
ax.text(0.5, 0.64, r'$\theta$', color='w', ha='center', va='center', fontsize=6.5)
ax.text(0.1, 0.72, '$i$', ha='right', va='center', fontsize=6.5); ax.text(0.1, 0.56, '$i{+}1$', ha='right', va='center', fontsize=6.5)
ax.text(0.5, 0.84, r'$U(\theta)=e^{-i\theta\,\mathbf{S}_i\cdot\mathbf{S}_{i+1}}$', ha='center', va='center', fontsize=8)
ax.text(0.5, 0.40, r'$\propto\cos\frac{\theta}{2}\,\mathbb{1}-i\sin\frac{\theta}{2}\,\mathrm{SWAP}$', ha='center', va='center', fontsize=7)
ax.text(0.5, 0.24, r'$\theta=0$: identity', ha='center', va='center', fontsize=7)
ax.text(0.5, 0.14, r'$\theta=\pi$: SWAP', ha='center', va='center', fontsize=7)
ax.text(0.5, 0.04, r'$\theta=\pi/2$: $\sqrt{\mathrm{SWAP}}$', ha='center', va='center', fontsize=7)
ax.text(0.5, -0.09, 'commutes with $\\mathbf{S}^2$, $S_z$', ha='center', va='center', fontsize=6.0, color='0.3')
# (b) two-spin S-T, N=6, n=3
panel(axes[1], 6, 4, 2, [r'$|S\rangle$ or $|T_0\rangle$'] * 3,
      ['$b_1$', '$b_2$', '$b_3$'], '(b) two-spin S–T code, $N=2n$ ($n=3$)', '')
# (c) three-spin EO, N=6, n=2
panel(axes[2], 6, 4, 3, [r'$|0_L\rangle$'] * 2, ['$b_1$', '$b_2$'],
      '(c) three-spin EO code, $N=3n$ ($n=2$)', 'not read')
axes[1].text(-3.0, -0.75, 'or Néel\n$|\\!\\uparrow\\downarrow\\uparrow\\downarrow\\cdots\\rangle$', ha='center', va='center', fontsize=6.3, color='0.3', linespacing=1.3)
axes[2].text(1.4, -0.75, '$|0_L\\rangle$: singlet on the first pair, $\\uparrow$ on the third dot\n$S_z=n/2$ sector', ha='center', va='center', fontsize=6.3, color='0.3', linespacing=1.3)
axes[1].text(2.0, -0.75, r'$S_z=0$ sector', ha='center', va='center', fontsize=6.3, color='0.3')
fig.tight_layout(pad=0.4, w_pad=0.8)
print('axes widths', [round(a.get_position().width, 3) for a in axes])
fig.savefig(OUT + 'fig_circuit.pdf'); fig.savefig(OUT + 'fig_circuit.png', dpi=160); plt.close(fig)

# ---- selection-rule picture: sectors S feeding Hamming classes k, two-spin code, n=3 (N=6) ----
n = 3; N = 2 * n
mult = [comb(N, N // 2 - S) - (comb(N, N // 2 - S - 1) if N // 2 - S - 1 >= 0 else 0) for S in range(n + 1)]      # multiplicity of S at S_z=0
neel = [m / comb(N, N // 2) for m in mult]                                           # Néel weights c_S^2 = m_S / C(N, N/2)
def Lam(k): return {1} if k == 1 else set(range(k + 1))
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for k in range(n + 1):
    for S in range(n + 1):
        ok = S in Lam(k)
        ax.add_patch(Rectangle((k - .45, S - .45), .9, .9, fc='#d8e4f4' if ok else 'white', ec='0.6', lw=0.5))
        if ok: ax.text(k, S, '●', ha='center', va='center', fontsize=7, color=BLUE)
        else: ax.text(k, S, '×', ha='center', va='center', fontsize=7, color='0.6')
ax.set_xticks(range(n + 1)); ax.set_xticklabels([f'$k={k}$' for k in range(n + 1)])
ax.set_yticks(range(n + 1)); ax.set_yticklabels([f'$S={S}$' for S in range(n + 1)])
ax.set_xlabel('PSB outcome class: $k=|b|$ triplet pairs'); ax.set_ylabel('total spin sector of $\\psi_0$')
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0, labelsize=6.5)
# weights of three initial states on the right (aligned columns)
cols_x = [n + .85, n + 1.65, n + 2.45]
ax.text(n + .85, n + .55, r'sector weights $c_S^2$ of $\psi_0$', fontsize=6, ha='left', va='center')
for x, h in zip(cols_x, [r'$|S\rangle^{\otimes3}$', r'$|ST_0S\rangle$', 'N\u00e9el']):
    ax.text(x, n + .12, h, fontsize=5.6, ha='center', va='center')
sing = [1, 0, 0, 0]; sts = [0, 1, 0, 0]          # |S T0 S>: one spin-1 between two singlets -> pure S=1
for S in range(n + 1):
    for x, v in zip(cols_x, [sing[S], sts[S], neel[S]]):
        ax.text(x, S - .25, f'{v:.2f}', fontsize=5.8, ha='center', va='center')
ax.set_xlim(-.6, n + 2.9); ax.set_ylim(-.6, n + .9)
fig.tight_layout(pad=0.3)
fig.savefig(OUT + 'fig_sectors.pdf', bbox_inches='tight'); fig.savefig(OUT + 'fig_sectors.png', dpi=160, bbox_inches='tight'); plt.close(fig)
print('neel weights', [round(x, 3) for x in neel], 'mult', mult)

# ---- beginner figure: the ZZ.SWAP network of superconducting hardware versus one exchange pulse ----
def wires(ax, ys, x0, x1, labels=None, lw=0.9):
    for k, y in enumerate(ys):
        ax.plot([x0, x1], [y, y], color=GREY, lw=lw, zorder=1)
        if labels: ax.text(x0 - .08, y, labels[k], ha='right', va='center', fontsize=7)

def box2(ax, x, y0, y1, w, text, fc, fs=6.5, ec='0.3'):
    """a box spanning two wires y0<y1"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y0 - .22), w, (y1 - y0) + .44, boxstyle='round,pad=0.02', fc=fc, ec=ec, lw=0.7, zorder=3))
    ax.text(x, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, zorder=4)

def box1(ax, x, y, w, text, fc='#f3f3f3'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - .17), w, .34, boxstyle='round,pad=0.02', fc=fc, ec='0.3', lw=0.6, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=5.8, zorder=4)

def cz(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color='k', lw=0.9, zorder=2)
    ax.plot([x, x], [y0, y1], 'o', color='k', ms=3.2, zorder=3)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.35, 1.1]))
# (a) the shared network shape
ax = axes[0]; ax.axis('off')
ax.set_title('(a) the network: parallel bubble sort', fontsize=7.5, loc='left')
n = 4; R = 4
wires(ax, range(n), -.3, R + .3, [f'$q_{i}$' for i in range(n)])
for r in range(R):
    for i in range(r % 2, n - 1, 2):
        box2(ax, r + .5, i, i + 1, .62, '', '#fff2cc', ec='0.4')
ax.text(-.3, n - .3, 'round', ha='right', va='bottom', fontsize=6.5, color='0.3')
for r in range(R): ax.text(r + .5, n - .3, f'{r+1}', ha='center', va='bottom', fontsize=6.5, color='0.3')
ax.text(R / 2, -1.0, 'one box = one two-qubit block\non a neighbouring pair; after $n$\nrounds every pair has met once\nand the order is reversed', ha='center', va='center', fontsize=6.8, color='0.25')
ax.set_xlim(-1.0, R + .6); ax.set_ylim(-2.0, n + .4)
# (b) superconducting block: ZZ(phi).SWAP = native gates, depth 4
ax = axes[1]; ax.axis('off')
ax.set_title('(b) superconducting qubits: ZZ$\\cdot$SWAP block', fontsize=7.5, loc='left')
wires(ax, [0, 1], 0, 1.3, ['$q_i$', '$q_{i+1}$'])
box2(ax, .65, 0, 1, 1.0, 'ZZ($\\varphi$)\nSWAP', '#fff2cc', fs=6.5)
ax.text(1.65, .5, '=', ha='center', va='center', fontsize=10)
wires(ax, [0, 1], 2.0, 6.3)
# ZZ module = CNOT, Rz(phi) on the target, CNOT; then a SWAP (Yan et al., depth 4 before native compilation)
def cnot(ax, x, yc, yt):
    ax.plot([x, x], [yc, yt], color='k', lw=0.9, zorder=2); ax.plot([x], [yc], 'o', color='k', ms=3.2, zorder=3)
    ax.add_patch(plt.Circle((x, yt), .17, fc='white', ec='k', lw=0.9, zorder=3)); ax.plot([x - .17, x + .17], [yt, yt], color='k', lw=0.9, zorder=4); ax.plot([x, x], [yt - .17, yt + .17], color='k', lw=0.9, zorder=4)
def swapg(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color='k', lw=0.9, zorder=2)
    for y in (y0, y1): ax.plot([x - .14, x + .14], [y - .14, y + .14], color='k', lw=0.9, zorder=3); ax.plot([x - .14, x + .14], [y + .14, y - .14], color='k', lw=0.9, zorder=3)
cnot(ax, 2.6, 1, 0); box1(ax, 3.5, 0, .7, '$R_z(\\varphi)$'); cnot(ax, 4.4, 1, 0); swapg(ax, 5.6, 0, 1)
ax.text(3.5, 1.45, 'ZZ($\\varphi$) module', ha='center', va='center', fontsize=6.3, color='0.3'); ax.text(5.6, 1.45, 'SWAP', ha='center', va='center', fontsize=6.3, color='0.3')
ax.annotate('', xy=(4.7, 1.3), xytext=(2.3, 1.3), arrowprops=dict(arrowstyle='-', lw=0.5, color='0.5'))
ax.annotate('', xy=(6.3, -.55), xytext=(2.1, -.55), arrowprops=dict(arrowstyle='<->', lw=0.6, color='0.3'))
ax.text(4.2, -.9, 'two CNOTs, $R_z$, then SWAP:\n4 gate layers per round', ha='center', va='center', fontsize=6.8, color='0.25')
ax.text(3.2, 2.05, 'interaction ZZ($\\varphi$) and routing SWAP are\nseparate; the SWAP is pure overhead', ha='center', va='center', fontsize=6.8, color='0.25')
ax.set_xlim(-.6, 6.8); ax.set_ylim(-1.4, 2.5)
# (c) exchange pulse: one pulse is the whole block
ax = axes[2]; ax.axis('off')
ax.set_title('(c) this work: one exchange pulse', fontsize=7.5, loc='left')
wires(ax, [0, 1], 0, 1.3, ['dot $i$', 'dot $i{+}1$'])
box2(ax, .65, 0, 1, .7, '$U(\\theta)$', '#d6e4f5', fs=7)
ax.text(1.65, .5, '=', ha='center', va='center', fontsize=10)
# pulse trace J(t)
t = np.linspace(0, 1, 200); J = np.exp(-((t - .5) / .27) ** 2)
x0, x1, yb = 2.1, 4.6, -.05
ax.plot([x0 - .1, x1 + .1], [yb, yb], color='0.3', lw=0.7); ax.plot([x0 - .1, x0 - .1], [yb, yb + 1.25], color='0.3', lw=0.7)
ax.fill_between(x0 + t * (x1 - x0), yb, yb + J * 1.05, color=BLUE, alpha=0.25, lw=0)
ax.plot(x0 + t * (x1 - x0), yb + J * 1.05, color=BLUE, lw=1.2)
ax.text((x0 + x1) / 2, yb - .32, 'shaded area $=\\theta=\\int J\\,dt$', ha='center', va='center', fontsize=7, color='#1a4a80')
ax.text(x1 + .15, yb + .05, '$t$', ha='left', va='center', fontsize=6.5); ax.text(x0 - .2, yb + 1.25, '$J(t)$', ha='right', va='center', fontsize=6.5)
ax.text(2.2, -.95, 'one pulse = the whole block:\ndepth 1 per round', ha='center', va='center', fontsize=6.8, color='0.25')
ax.text(2.6, 1.95, 'interaction and routing are the same pulse:\n$\\theta=\\pi$ is a SWAP, $\\theta\\notin\\pi\\mathbb{Z}$ a partial SWAP\n(price: $U$ conserves total spin, Sec. III)', ha='center', va='center', fontsize=6.8, color='0.25')
ax.set_xlim(-.9, 5.0); ax.set_ylim(-1.4, 2.5)
fig.tight_layout(pad=0.4, w_pad=0.5)
fig.savefig(OUT + 'fig_zzswap.pdf'); fig.savefig(OUT + 'fig_zzswap.png', dpi=160); plt.close(fig)
