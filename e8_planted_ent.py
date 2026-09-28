"""E8d: 漏れなし かつ もつれる 論理ユニタリを深さ R で探す（ターゲット指定なし）。
損失 = 漏れ + λ·top1（1|2 切れ目の演算子 Schmidt 最大項）。見つかれば、それを植え込みターゲットに復元を試す。"""
import json, sys, time
import numpy as np, torch
from scipy.linalg import polar
from eo_net import logical_basis, bubble_schedule, init_thetas
from e8_polar_sector import sector, perms, block, compile_Q
torch.set_num_threads(4)
n = 3; R = int(sys.argv[1]); lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
B, N = logical_basis('eo', n); d = 8; idx, pos, Bs = sector(B, N); sched = bubble_schedule(N, R); P = perms(sched, idx, pos, N)
def top1(M):
    T = M.reshape(2, 4, 2, 4).permute(0, 2, 1, 3).reshape(4, 16); s = torch.linalg.svdvals(T); return (s[0] ** 2) / (s ** 2).sum()
out = []
for plant in range(4):
    th = init_thetas(sched, 'random', gen=torch.Generator().manual_seed(9500 + plant)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(400):
        opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (M.abs() ** 2).sum() / d + lam * top1(M); loss.backward(); opt.step()
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=600, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (M.abs() ** 2).sum() / d + lam * top1(M); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): M0 = block(th, P, Bs).numpy(); leak0 = 1 - (np.abs(M0) ** 2).sum() / d; t1 = float(top1(torch.tensor(M0)))
    print(f"R={R} lam={lam} plant {plant}: leak {leak0:.1e} top1 {t1:.3f}", flush=True)
    rec = dict(R=R, lam=lam, plant=plant, leak0=float(leak0), top1=t1)
    if leak0 < 1e-6:
        Q0, _ = polar(M0); t0 = time.time(); (F, L, M), Fs = compile_Q(Q0, sched, Bs, P, d, seeds=6)
        rec.update(F_proc=F, leak=L, F_all=Fs); print(f"R={R} plant {plant}: recovery F={F:.6f} leak={L:.1e} F_seeds[min/med]={min(Fs):.3f}/{np.median(Fs):.3f} [{time.time()-t0:.0f}s]", flush=True)
    out.append(rec); json.dump(out, open(f'results/e8_planted_ent_R{R}.json', 'w'), indent=1)
