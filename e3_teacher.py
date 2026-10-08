"""E3a: teacher distributions from GPT-2 small.
Restricted-vocabulary next-token chains: after a prompt, take the top-V next tokens as
vocabulary; p(t1..tL) = prod_k p_GPT2(t_k | prompt, t_<k) renormalised over V at every step.
V=8, L=2 -> 64 outcomes (6 bits); V=4, L=3 -> 64 outcomes (6 bits)."""
import sys, json, math, itertools
import numpy as np, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
torch.set_num_threads(4)

tok = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()

def next_probs(ids):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1]
    return torch.softmax(logits, -1)

def teacher(prompt, V, L):
    ids0 = tok.encode(prompt)
    p0 = next_probs(ids0)
    vocab = torch.topk(p0, V).indices.tolist()
    words = [tok.decode([v]) for v in vocab]
    dist = {}
    def rec(ids, prefix, logp):
        if len(prefix) == L:
            dist[tuple(prefix)] = math.exp(logp); return
        p = next_probs(ids)
        pv = p[vocab]; pv = pv / pv.sum()
        for k, v in enumerate(vocab):
            rec(ids + [v], prefix + [k], logp + math.log(float(pv[k])))
    rec(ids0, [], 0.0)
    P = np.array([dist[key] for key in itertools.product(range(V), repeat=L)])
    return words, P

if __name__ == '__main__':
    out = {}
    import os
    if os.path.exists('results/e3_teachers.json'): out = json.load(open('results/e3_teachers.json'))
    extra = [('once_V8L2', 'Once upon a', 8, 2), ('sky_V4L3', 'The sky above the city was', 4, 3),
             ('river_V8L2', 'The river flows', 8, 2), ('door_V4L3', 'She opened the door and', 4, 3)]
    base = [('cat_V8L2', 'The cat sat on', 8, 2), ('begin_V8L2', 'In the beginning', 8, 2), ('cat_V4L3', 'The cat sat on', 4, 3), ('begin_V4L3', 'In the beginning', 4, 3)]
    todo = extra if sys.argv[1:] == ['extra'] else base
    if len(sys.argv) >= 5 and sys.argv[1] not in ('extra',): todo = [(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))]
    for name, prompt, V, L in todo:
        words, P = teacher(prompt, V, L)
        H = -(P * np.log2(P + 1e-30)).sum()
        print(f"{name}: prompt={prompt!r} vocab={words}  H={H:.3f} bits (max {math.log2(len(P)):.1f})  max p={P.max():.3f}")
        out[name] = dict(prompt=prompt, V=V, L=L, vocab=words, p=P.tolist())
    json.dump(out, open('results/e3_teachers.json', 'w'), indent=1)
