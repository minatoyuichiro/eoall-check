"""
Exchange-only spin-chain networks (bubble-sort / staircase schedules) for
2-spin S-T units and 3-spin EO qubits.  torch, state-vector, autograd.

Conventions
-----------
* N physical spins, state tensor of shape (2,)*N, index 0 = up.
* Exchange pulse on adjacent spins (i,i+1) with pulse area a = J*tau:
      U(a) = exp(-i a S_i.S_j) = e^{i a/4} [cos(a/2) I - i sin(a/2) SWAP]
  a = pi  ->  SWAP (up to phase),  a = 0 -> identity.
* Schedules are lists of rounds; each round is a list of (i, i+1) pairs that
  are applied in parallel (disjoint).  Parameters: one pulse area per pair per
  round.
* Encodings
    'st' : 2-spin unit, logical |0> = singlet S, |1> = T0            (S_z = 0)
    'eo' : 3-spin unit, logical |0> = |S>_{12}|up>_3,
                        |1> = (2|uud> - |udu> - |duu>)/sqrt(6)       (S=1/2, m=+1/2)
  Readout for both: Pauli spin blockade on the first pair of each block,
  singlet -> bit 0, triplet -> bit 1.  Well defined also on leaked states.
"""
import math
import numpy as np
import torch

CDT = torch.complex128
RDT = torch.float64

# ---------------------------------------------------------------- gates
_SWAP = torch.tensor([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=CDT)
_I4 = torch.eye(4, dtype=CDT)


def exchange_gate(a):
    """4x4 unitary exp(-i a S.S) for pulse area a (torch scalar, differentiable)."""
    a = a.to(RDT)
    ph = torch.exp(1j * a / 4)
    return ph * (torch.cos(a / 2) * _I4 - 1j * torch.sin(a / 2) * _SWAP)


def apply_2site(psi, G, i, N):
    """Apply 4x4 gate G on adjacent sites (i, i+1) of state psi (shape (2,)*N)."""
    left = 2 ** i
    right = 2 ** (N - i - 2)
    x = psi.reshape(left, 4, right)
    x = torch.einsum('ab,xbz->xaz', G, x)
    return x.reshape((2,) * N)


# ------------------------------------------------------------ schedules
def bubble_schedule(N, rounds):
    """Parallel bubble sort (odd-even transposition) brickwork.
    rounds == N covers every pair exactly once and reverses the chain at a=pi."""
    sched = []
    for r in range(rounds):
        sched.append([(i, i + 1) for i in range(r % 2, N - 1, 2)])
    return sched


def staircase_schedule(N, layers):
    """Sequential (Ran-2020 style) staircase: (0,1),(1,2),...,(N-2,N-1) per layer."""
    return [[(i, i + 1)] for _ in range(layers) for i in range(N - 1)]


def n_params(sched):
    return sum(len(r) for r in sched)


def run_network(psi0, thetas, sched, N):
    """thetas: 1-D tensor with one entry per (round, pair) in schedule order."""
    psi = psi0
    k = 0
    for rnd in sched:
        for (i, _) in rnd:
            psi = apply_2site(psi, exchange_gate(thetas[k]), i, N)
            k += 1
    return psi


# ------------------------------------------------------------ encodings
def _ket(bits):
    v = torch.zeros((2,) * len(bits), dtype=CDT)
    v[tuple(bits)] = 1.0
    return v


def unit_states(enc):
    """Return (block_size, [|0_L>, |1_L>]) as tensors of shape (2,)*block."""
    if enc == 'st':
        S = (_ket([0, 1]) - _ket([1, 0])) / math.sqrt(2)
        T0 = (_ket([0, 1]) + _ket([1, 0])) / math.sqrt(2)
        return 2, [S, T0]
    if enc == 'eo':
        z = (_ket([0, 1, 0]) - _ket([1, 0, 0])) / math.sqrt(2)          # |S>_12 |up>
        o = (2 * _ket([0, 0, 1]) - _ket([0, 1, 0]) - _ket([1, 0, 0])) / math.sqrt(6)
        return 3, [z, o]
    raise ValueError(enc)


def logical_basis(enc, n):
    """Matrix B of shape (2^N, 2^n): columns = logical product states."""
    b, u = unit_states(enc)
    N = b * n
    cols = []
    for x in range(2 ** n):
        bits = [(x >> (n - 1 - j)) & 1 for j in range(n)]
        v = u[bits[0]]
        for j in range(1, n):
            v = torch.tensordot(v, u[bits[j]], dims=0)
        cols.append(v.reshape(-1))
    return torch.stack(cols, dim=1), N


def embed_logical(enc, n, c):
    """c: (2^n,) logical amplitudes -> physical state tensor (2,)*N."""
    B, N = logical_basis(enc, n)
    return (B @ c.to(CDT)).reshape((2,) * N)


def product_state(enc, n, which='0'):
    """Initial product states.  'st': '0'=|S..S>, '1'=|T0..T0>, 'alt'=|S T0 S T0..>.
    'eo': '0'=|0..0>, '1'=|1..1>."""
    b, u = unit_states(enc)
    if which == 'neel':                  # every block in (|0_L>+|1_L>)/sqrt2; for 'st' this is |up down> per pair
        c = torch.ones(2 ** n, dtype=CDT) / math.sqrt(2 ** n)
        return embed_logical(enc, n, c)
    pat = {'0': [0] * n, '1': [1] * n, 'alt': [j % 2 for j in range(n)]}.get(which)
    if pat is None:                      # explicit bit pattern, e.g. '011011'
        assert len(which) == n and set(which) <= {'0', '1'}, which
        pat = [int(ch) for ch in which]
    c = torch.zeros(2 ** n, dtype=CDT)
    idx = int(''.join(str(p) for p in pat), 2)
    c[idx] = 1.0
    return embed_logical(enc, n, c)


def leakage(psi, enc, n):
    """1 - weight inside the logical (code) subspace."""
    B, N = logical_basis(enc, n)
    amp = B.conj().T @ psi.reshape(-1)
    return 1.0 - (amp.abs() ** 2).sum().real


# -------------------------------------------------------------- readout
# 2-spin basis change: row 0 = <S|, rest orthonormal completion
_V_PAIR = torch.tensor([[0, 1, -1, 0],
                        [0, 1, 1, 0],
                        [math.sqrt(2), 0, 0, 0],
                        [0, 0, 0, math.sqrt(2)]], dtype=CDT) / math.sqrt(2)


def psb_probs(psi, enc, n):
    """Pauli-spin-blockade readout on the first pair of every block.
    Returns p of shape (2,)*n : bit 0 = singlet, 1 = any triplet."""
    b = 2 if enc == 'st' else 3
    N = b * n
    x = psi
    for j in range(n):
        x = apply_2site(x, _V_PAIR, b * j, N)
    q = (x.abs() ** 2)
    # reshape to (block0: 4, rest0 ; block1: 4, rest1 ; ...)
    if enc == 'st':
        q = q.reshape((4,) * n)
    else:
        q = q.reshape(*([4, 2] * n)).sum(dim=tuple(range(1, 2 * n, 2)))
    # singlet index 0 vs triplet 1..3
    p = q
    for j in range(n):
        s = p.narrow(j, 0, 1)
        t = p.narrow(j, 1, 3).sum(dim=j, keepdim=True)
        p = torch.cat([s, t], dim=j)
    return p


def kl(p_teacher, p_model, eps=1e-12):
    pt = p_teacher.reshape(-1)
    pm = p_model.reshape(-1)
    return (pt * (torch.log(pt + eps) - torch.log(pm + eps))).sum()


# -------------------------------------------------------------- helpers
def init_thetas(sched, mode, eps=0.1, gen=None):
    m = n_params(sched)
    g = gen or torch.Generator().manual_seed(0)
    if mode == 'random':
        return (2 * math.pi * torch.rand(m, generator=g, dtype=RDT)).requires_grad_(True)
    base = {'zero': 0.0, 'pi': math.pi}[mode]
    return (base + eps * torch.randn(m, generator=g, dtype=RDT)).requires_grad_(True)


def haar_logical(n, gen):
    c = torch.randn(2 ** n, generator=gen, dtype=RDT) + 1j * torch.randn(2 ** n, generator=gen, dtype=RDT)
    return (c / torch.linalg.norm(c)).to(CDT)


def random_mps_logical(n, chi, gen):
    """Random open-boundary MPS with bond dim chi over n qubits -> (2^n,) amplitudes."""
    ts = []
    for j in range(n):
        dl = 1 if j == 0 else chi
        dr = 1 if j == n - 1 else chi
        t = torch.randn(dl, 2, dr, generator=gen, dtype=RDT) + 1j * torch.randn(dl, 2, dr, generator=gen, dtype=RDT)
        ts.append(t.to(CDT))
    v = ts[0]
    for t in ts[1:]:
        v = torch.einsum('...a,abc->...bc', v, t)
    c = v.reshape(-1)
    return c / torch.linalg.norm(c)


def fidelity(psi, target):
    ov = torch.vdot(target.reshape(-1), psi.reshape(-1))
    return (ov.abs() ** 2).real
