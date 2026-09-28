"""E8b: 極分解の直交因子 Q の演算子 Schmidt スペクトル（GPT-2 c_proj の d×d 角、二分割 d = da*db）vs Haar 直交。
Schmidt 分解と組み合わせて Q を短くできるかの判定材料。"""
import json, numpy as np
from scipy.linalg import polar
from scipy.stats import ortho_group
from transformers import GPT2LMHeadModel
m = GPT2LMHeadModel.from_pretrained('gpt2')
def op_schmidt(Q, da, db):
    # Q (da*db x da*db) -> 再配置して (da^2 x db^2) の特異値
    T = Q.reshape(da, db, da, db).transpose(0, 2, 1, 3).reshape(da * da, db * db)
    s = np.linalg.svd(T, compute_uv=False); w = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(w)
    return dict(top1=float(w[0]), n90=int(np.searchsorted(cum, 0.9) + 1), n99=int(np.searchsorted(cum, 0.99) + 1),
                rank_max=int(min(da * da, db * db)), entropy_bits=float(-(w[w > 1e-15] * np.log2(w[w > 1e-15])).sum()))
out = []
for d, da, db in [(4, 2, 2), (8, 2, 4), (16, 4, 4), (64, 8, 8), (256, 16, 16)]:
    for layer in [0, 6, 11]:
        for kind in ['attn.c_proj', 'mlp.c_fc']:
            mod = m.transformer.h[layer].attn.c_proj if kind == 'attn.c_proj' else m.transformer.h[layer].mlp.c_fc
            W = mod.weight.detach().numpy()[:d, :d]; Q, P = polar(W)
            r = op_schmidt(Q, da, db); r.update(d=d, layer=layer, kind=kind, src='gpt2',
                                               P_dev=float(np.linalg.norm(P - np.eye(d)) / np.linalg.norm(P)),
                                               W_orth_dev=float(np.linalg.norm(W - Q) / np.linalg.norm(W)))
            out.append(r)
    for s in range(3):
        Q = ortho_group.rvs(d, random_state=s); r = op_schmidt(Q, da, db); r.update(d=d, layer=s, kind='haar', src='haar'); out.append(r)
    # 参考：ランダム正規行列の極分解 Q（E4 のターゲットと同種）
    for s in range(3):
        W = np.random.default_rng(s).standard_normal((d, d)); Q, _ = polar(W); r = op_schmidt(Q, da, db); r.update(d=d, layer=s, kind='gauss_polar', src='gauss'); out.append(r)
json.dump(out, open('results/e8_schmidt_q.json', 'w'), indent=1)
for r in out: print(f"d={r['d']:3d} {r['src']:5s} {r['kind']:12s} L{r['layer']:2d} top1={r['top1']:.3f} n90={r['n90']:3d}/{r['rank_max']:3d} n99={r['n99']:3d} S={r['entropy_bits']:.2f}b" + (f" |P-I|/|P|={r['P_dev']:.2f}" if 'P_dev' in r else ''))
