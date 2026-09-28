import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

np.random.seed(0)

# --- spin-1/2 operators on 4 spins (16-dim) ---
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)

def op(single, site, n=4):
    mats = [I2] * n
    mats[site] = single
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def heis(i, j):
    return sum(op(s, i) @ op(s, j) for s in (sx, sy, sz))

H12 = heis(0, 1)
H23 = heis(1, 2)
H34 = heis(2, 3)

# --- logical basis: pair A = spins(0,1), pair B = spins(2,3) ---
up = np.array([1, 0], dtype=complex)
dn = np.array([0, 1], dtype=complex)

def kron_all(vs):
    out = vs[0]
    for v in vs[1:]:
        out = np.kron(out, v)
    return out

S_pair  = (kron_all([up, dn]) - kron_all([dn, up])) / np.sqrt(2)
T0_pair = (kron_all([up, dn]) + kron_all([dn, up])) / np.sqrt(2)

logical = [np.kron(a, b) for a in (S_pair, T0_pair) for b in (S_pair, T0_pair)]
B = np.stack(logical, axis=1)   # 16x4, columns = |SS>,|ST0>,|T0S>,|T0T0>
P = B @ B.conj().T              # projector onto logical subspace

def target(theta):
    # logical ZZ(theta) * pair-SWAP,  basis order SS, ST0, T0S, T0T0
    T = np.zeros((4, 4), dtype=complex)
    T[0, 0] = np.exp(-1j * theta)
    T[3, 3] = np.exp(-1j * theta)
    T[2, 1] = np.exp(+1j * theta)   # ST0 -> T0S
    T[1, 2] = np.exp(+1j * theta)   # T0S -> ST0
    return T

def sequence_unitary(params, pattern):
    """pattern: list of 'A' (H12 & H34, two angles) or 'B' (H23, one angle)."""
    U = np.eye(16, dtype=complex)
    k = 0
    for step in pattern:
        if step == 'A':
            a, c = params[k], params[k + 1]
            k += 2
            U = expm(-1j * (a * H12 + c * H34)) @ U
        else:
            b = params[k]
            k += 1
            U = expm(-1j * b * H23) @ U
    return U

def nparams(pattern):
    return sum(2 if s == 'A' else 1 for s in pattern)

def infidelity(params, pattern, T):
    U = sequence_unitary(params, pattern)
    M = B.conj().T @ U @ B          # 4x4 logical block
    F = abs(np.trace(T.conj().T @ M)) / 4.0
    return 1.0 - F

def leakage(params, pattern):
    U = sequence_unitary(params, pattern)
    M = B.conj().T @ U @ B
    return 1.0 - np.linalg.norm(M, 'fro')**2 / 4.0   # 0 if logical subspace preserved

def search(pattern, theta, restarts=40, x0=None):
    T = target(theta)
    best = None
    if x0 is not None:
        best = minimize(infidelity, x0, args=(pattern, T), method='BFGS',
                        options={'maxiter': 5000, 'gtol': 1e-14})
        if best.fun < 1e-10:
            return best
    for r in range(restarts):
        xr = np.random.uniform(0, 2 * np.pi, nparams(pattern))
        res = minimize(infidelity, xr, args=(pattern, T), method='BFGS',
                       options={'maxiter': 2000, 'gtol': 1e-12})
        if best is None or res.fun < best.fun:
            best = res
        if best.fun < 1e-10:
            break
    return best

# 9-step pattern: A = simultaneous (J12, J34), B = J23. 14 parameters.
# This matches the dimension of the reachable group u(2)+u(3)+u(1) under
# Schur-Weyl duality; 7 steps (11 params) plateaus at infidelity ~0.076.
PAT9 = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A']

# Known solution for theta = pi/4 (angles J*tau mod 2pi), found 2026-08:
X_PI4 = np.array([
    4.3141, 4.3893,           # step1 A: J12, J34
    3.6378,                   # step2 B: J23
    2.0970, 4.6482,           # step3 A
    4.7693,                   # step4 B
    3.0205, 3.2627,           # step5 A
    1.5139,                   # step6 B
    1.6350, 4.1862,           # step7 A
    2.6454,                   # step8 B
    3.4647, 0.3983,           # step9 A
])

if __name__ == '__main__':
    theta = np.pi / 4
    print("=== target: logical ZZ(theta) * pair-SWAP, theta = pi/4 ===")
    res = search(PAT9, theta, x0=X_PI4)
    print(f"9-step sequence: infidelity={res.fun:.3e}  leakage={leakage(res.x, PAT9):.3e}")
    np.save('best_x.npy', res.x)

    print("\n=== theta sweep (continuous family, warm-started) ===")
    xprev = res.x
    for th in [0.1, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2.0, 2.8]:
        r = search(PAT9, th, restarts=60, x0=xprev)
        xprev = r.x
        print(f"theta={th:.3f}  infidelity={r.fun:.3e}  leakage={leakage(r.x, PAT9):.3e}")

    print("\n=== pulse angles (J*tau mod 2pi) for theta=pi/4 ===")
    x = np.load('best_x.npy')
    k = 0
    for i, step in enumerate(PAT9):
        if step == 'A':
            print(f" step{i+1} A: J12t={x[k]%(2*np.pi):.6f}  J34t={x[k+1]%(2*np.pi):.6f}")
            k += 2
        else:
            print(f" step{i+1} B: J23t={x[k]%(2*np.pi):.6f}")
            k += 1
