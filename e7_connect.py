"""E7: is the 1-D bubble-sort (odd-even) brickwork the right way to get all-to-all
coupling at low depth?  Compare, at equal depth R, schedules of disjoint exchange
pulses on
  bubble     : 1-D chain, odd-even transposition (covers every pair once at R=N)
  grid       : 2-D grid (blocks laid out along rows), rounds cycle row-even /
               row-odd / col-even / col-odd
  roundrobin : non-local exact cover of K_N (circle method), N-1 rounds cover
               every pair once; no geometry at all
  randmatch  : a fresh random perfect matching of all N spins every round
Lower bound for any of them: a spin meets at most one partner per round, so
every pair meets only after >= N-1 rounds, in any dimension.
Same teacher / encoding / readout / loss as E3."""
import json, math, sys, time, itertools
import numpy as np, torch
from eo_net import *


def apply_pair(psi, G, i, j, N):
    """Apply 4x4 gate G on arbitrary sites (i, j) of psi (shape (2,)*N)."""
    if j == i + 1:
        return apply_2site(psi, G, i, N)
    if i == j + 1:
        return apply_2site(psi, G, j, N)
    x = torch.movedim(psi, (i, j), (0, 1)).reshape(4, -1)
    x = G @ x
    x = x.reshape((2, 2) + (2,) * (N - 2))
    return torch.movedim(x, (0, 1), (i, j))


def run_general(psi0, thetas, sched, N):
    psi = psi0; k = 0
    for rnd in sched:
        for (i, j) in rnd:
            psi = apply_pair(psi, exchange_gate(thetas[k]), i, j, N)
            k += 1
    return psi


# ------------------------------------------------------------------ schedules
def grid_schedule(N, b, rounds):
    """2-D grid with rows of length Lx (a multiple of the block size b), snake-free
    row-major layout so that block j = sites b*j..b*j+b-1 lies inside one row."""
    Lx = {12: 4, 16: 4, 18: 6, 24: 6}[N]
    Ly = N // Lx
    assert Lx % b == 0 and Lx * Ly == N
    idx = lambda r, c: r * Lx + c
    row_even = [(idx(r, c), idx(r, c + 1)) for r in range(Ly) for c in range(0, Lx - 1, 2)]
    row_odd = [(idx(r, c), idx(r, c + 1)) for r in range(Ly) for c in range(1, Lx - 1, 2)]
    col_even = [(idx(r, c), idx(r + 1, c)) for c in range(Lx) for r in range(0, Ly - 1, 2)]
    col_odd = [(idx(r, c), idx(r + 1, c)) for c in range(Lx) for r in range(1, Ly - 1, 2)]
    cyc = [row_even, col_even, row_odd, col_odd]
    cyc = [m for m in cyc if m]
    return [cyc[r % len(cyc)] for r in range(rounds)]


def roundrobin_schedule(N, rounds):
    """Circle method: N-1 perfect matchings covering every pair of K_N exactly once."""
    assert N % 2 == 0
    ms = []
    ring = list(range(N - 1))
    for r in range(N - 1):
        m = [(N - 1, ring[0])]
        for k in range(1, N // 2):
            m.append((ring[k], ring[-k]))
        ms.append([tuple(sorted(p)) for p in m])
        ring = [ring[-1]] + ring[:-1]
    return [ms[r % (N - 1)] for r in range(rounds)]


def randmatch_schedule(N, rounds, seed=0):
    g = np.random.default_rng(seed)
    sched = []
    for _ in range(rounds):
        perm = g.permutation(N)
        sched.append([tuple(sorted((int(perm[2 * k]), int(perm[2 * k + 1])))) for k in range(N // 2)])
    return sched


def make_schedule(kind, N, b, R):
    if kind == 'bubble':
        return bubble_schedule(N, R)
    if kind == 'grid':
        return grid_schedule(N, b, R)
    if kind == 'roundrobin':
        return roundrobin_schedule(N, R)
    if kind == 'randmatch':
        return randmatch_schedule(N, R)
    raise ValueError(kind)


def coverage(sched, N):
    """fraction of the N(N-1)/2 pairs that meet at least once."""
    met = set()
    for rnd in sched:
        for p in rnd:
            met.add(tuple(sorted(p)))
    return len(met) / (N * (N - 1) // 2)


# ------------------------------------------------------------------ training
def train_kl(psi0, th, sched, N, enc, n, pt, steps, lr=0.05):
    opt = torch.optim.Adam([th], lr=lr)
    best = (1e9, None); hist = []
    for it in range(steps):
        opt.zero_grad()
        p = psb_probs(run_general(psi0, th, sched, N), enc, n)
        loss = kl(pt, p)
        loss.backward()
        hist.append(loss.item())
        if loss.item() < best[0]:
            best = (loss.item(), th.detach().clone())   # parameters that produced this loss
        opt.step()
    return best, hist


def main(enc, tname, init_state, kind, R, mode, steps, threads, seed=0):
    torch.set_num_threads(threads)
    T = json.load(open('results/e3_teachers.json'))[tname]
    n = int(math.log2(len(T['p'])))
    b = 2 if enc == 'st' else 3; N = b * n
    pt = torch.tensor(T['p'], dtype=RDT).reshape((2,) * n)
    sched = make_schedule(kind, N, b, R)
    psi0 = product_state(enc, n, init_state)
    g = torch.Generator().manual_seed(seed)
    th = init_thetas(sched, mode, gen=g)
    rec = dict(enc=enc, teacher=tname, n=n, N=N, init_state=init_state, kind=kind, R=R,
               params=n_params(sched), coverage=coverage(sched, N), init=mode, steps=steps, seed=seed)
    t0 = time.time()
    with torch.no_grad():
        rec['kl_init'] = float(kl(pt, psb_probs(run_general(psi0, th, sched, N), enc, n)))
    (best_kl, th_best), hist = train_kl(psi0, th, sched, N, enc, n, pt, steps)
    with torch.no_grad():
        psi = run_general(psi0, th_best, sched, N)
        p = psb_probs(psi, enc, n)
        rec['leak_final'] = float(leakage(psi, enc, n))
        rec['tv_final'] = float(0.5 * (p - pt).abs().sum())
    rec['kl_final'] = best_kl
    rec['hist'] = hist[::10]
    rec['thetas'] = th_best.tolist()
    rec['time_s'] = time.time() - t0
    print(f"{enc} {tname} |{init_state}> {kind} R={R} p={rec['params']} cov={rec['coverage']:.2f} {mode}: "
          f"KL {rec['kl_init']:.4f} -> {best_kl:.4f}  TV={rec['tv_final']:.4f} leak={rec['leak_final']:.3f} [{rec['time_s']:.0f}s]", flush=True)
    return rec


if __name__ == '__main__':
    enc, tname, init_state, kind, R, mode = sys.argv[1:7]
    R = int(R)
    steps = int(sys.argv[7]) if len(sys.argv) > 7 else 400
    threads = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    seed = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    rec = main(enc, tname, init_state, kind, R, mode, steps, threads, seed)
    tag = f"{enc}_{tname}_{init_state}_{kind}{R}_{mode}" + (f"_seed{seed}" if seed else "")
    json.dump(rec, open(f'results/e7_{tag}.json', 'w'))
