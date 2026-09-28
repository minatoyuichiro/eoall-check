"""E3b: fit MPS Born machines (bond dim chi) to the GPT-2 teacher distributions by KL.
Saves p_MPS and the (normalised) MPS amplitudes for later state-level pretraining."""
import json, math
import numpy as np, torch
torch.set_num_threads(4)
RDT = torch.float64

def mps_amplitudes(ts):
    v = ts[0]
    for t in ts[1:]:
        v = torch.einsum('...a,abc->...bc', v, t)
    return v.reshape(-1)

def fit(p_teacher, n, chi, steps=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ts = []
    for j in range(n):
        dl = 1 if j == 0 else chi; dr = 1 if j == n - 1 else chi
        t = (torch.randn(dl, 2, dr, generator=g, dtype=RDT) * 0.5).requires_grad_(True)
        ts.append(t)
    pt = torch.tensor(p_teacher, dtype=RDT)
    opt = torch.optim.Adam(ts, lr=0.02)
    best = (1e9, None)
    for it in range(steps):
        opt.zero_grad()
        c = mps_amplitudes(ts)
        p = c ** 2; p = p / p.sum()
        loss = (pt * (torch.log(pt + 1e-12) - torch.log(p + 1e-12))).sum()
        loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), [t.detach().clone() for t in ts])
    c = mps_amplitudes(best[1]); c = c / c.norm()
    return best[0], c.detach()

if __name__ == '__main__':
    T = json.load(open('results/e3_teachers.json'))
    out = {}
    for name, d in T.items():
        n = int(math.log2(len(d['p'])))
        out[name] = {}
        for chi in [1, 2, 3, 4, 8]:
            kls = []; cs = []
            for seed in range(3):
                kl, c = fit(d['p'], n, chi, seed=seed); kls.append(kl); cs.append(c)
            k = int(np.argmin(kls)); c = cs[k]
            out[name][f'chi{chi}'] = dict(kl=kls[k], kl_seeds=kls, amp=c.numpy().tolist(), p=(c.numpy() ** 2).tolist())
            print(f"{name} chi={chi}: KL(teacher||MPS) = {kls[k]:.4f} nats  (seeds {np.round(kls,4)})", flush=True)
    json.dump(out, open('results/e3_mps.json', 'w'))
