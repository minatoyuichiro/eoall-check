"""E8c: 植え込みターゲット。深さ R の網で「漏れなし」になる θ を見つけ（ターゲットは指定しない）、
その論理ユニタリ Q0（= 構成により深さ R で到達可能）を、別の乱数初期化から復元できるかを試す。
復元できれば Haar/GPT-2 ターゲットの失敗は表現力（深さ不足）、できなければ最適化の問題。"""
import json, sys, time
import numpy as np, torch
from eo_net import logical_basis, bubble_schedule, n_params, init_thetas, CDT
from e8_polar_sector import sector, perms, block, compile_Q
torch.set_num_threads(4)
n = 3; R = int(sys.argv[1]); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
B, N = logical_basis('eo', n); d = 8; idx, pos, Bs = sector(B, N); sched = bubble_schedule(N, R); P = perms(sched, idx, pos, N)
out = []
for plant in range(2):
    th = init_thetas(sched, 'random', gen=torch.Generator().manual_seed(9000 + plant)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(300):
        opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (M.abs() ** 2).sum() / d; loss.backward(); opt.step()
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=600, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); M = block(th, P, Bs); loss = 1 - (M.abs() ** 2).sum() / d; loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad(): M0 = block(th, P, Bs).numpy(); leak0 = 1 - (np.abs(M0) ** 2).sum() / d
    from scipy.linalg import polar as _polar
    Q0, _ = _polar(M0)  # 漏れがほぼ 0 なら M0 はユニタリ; 最近接ユニタリに丸める（QR は位相を変えるので不可）
    T = Q0.reshape(2, 4, 2, 4).transpose(0, 2, 1, 3).reshape(4, 16); s = np.linalg.svd(T, compute_uv=False); top1 = float(s[0] ** 2 / (s ** 2).sum())
    print(f"R={R} plant {plant}: leak of planted network {leak0:.1e}, |M0-Q0|={np.abs(M0-Q0).max():.1e}, op-Schmidt top1 (1|2 cut) {top1:.3f}", flush=True)
    t0 = time.time(); (F, L, M), Fs = compile_Q(Q0, sched, Bs, P, d, seeds=seeds)
    rec = dict(R=R, plant=plant, leak0=float(leak0), top1=top1, F_proc=F, leak=L, F_all=Fs); out.append(rec)
    print(f"R={R} plant {plant}: recovery F={F:.6f} leak={L:.1e} F_seeds[min/med]={min(Fs):.3f}/{np.median(Fs):.3f} [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(f'results/e8_planted_R{R}.json', 'w'), indent=1)
