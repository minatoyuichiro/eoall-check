# Bubble-sort exchange networks — numerics

Code and raw results (release `v1`, matching the arXiv submission) for the manuscript *Bubble-Sort Exchange Networks: Symmetry Selection Rules and
Language-Model Distillation on Exchange-Only Spin Chains* (Y. Minato, blueqat).

Everything runs on a CPU with `numpy`, `scipy`, `torch` (CPU build) and, for the GPT-2 teachers,
`transformers`. All results in `results/` were produced on a single 32-core machine; the `.log` files are
the console output of the corresponding script and the `.json` files hold the numbers quoted in the paper.

## Layout

| file | what it does | paper |
|---|---|---|
| `eo_net.py` | common core: state-vector simulator, exchange pulse `U(θ)=e^{iθ/4}[cos(θ/2) 1 − i sin(θ/2) SWAP]`, bubble-sort / staircase schedules, `st` (two-spin S–T) and `eo` (three-spin) encodings, Pauli-spin-blockade readout, initial states | Sec. II |
| `e2_reach.py`, `e2b_ceiling.py`, `e2c_depth.py` | state preparation vs. depth; SU(2) fidelity ceiling `(Σ_S ‖P_S t‖‖P_S ψ0‖)²` | Sec. III.A, Table I |
| `e3_teacher.py` → `e3_mps.py` → `e3_eo.py` | GPT-2 restricted-vocabulary teachers → MPS Born-machine fits → end-to-end distillation into the exchange network; `run_e3_*.sh` are the sweeps | Sec. IV, Table III |
| `e3d_dist_ceiling.py`, `e3e_selection_rule.py` | distribution-level ceiling by direct optimisation over the symmetry-allowed set, and the PSB selection rule / TV floor | Sec. III.B–C, Table II |
| `e4_unitary.py` | compilation of logical unitaries (polar factors) — full 2^N state vector, slow | Appendix A |
| `e8_polar_sector.py`, `e8_planted.py`, `e8_planted_ent.py` | the same in the fixed-`S_z` sector (84 amplitudes at N=9), multi-restart Adam+L-BFGS, and the planted-target test that separates expressivity from optimisation | Appendix A |
| `e8_schmidt_q.py` | operator-Schmidt spectra of the polar factors of GPT-2 weight blocks vs. Haar | Appendix A |
| `run_e7_seeds.sh`, `e7_table_seeds.py` | seed statistics for the layout comparison (Table IV: median and range over seeds) | Sec. IV.D |
| `run_review_extra.sh`, `e3_teacher.py extra`, `e3_mps.py <teachers>`, `e3e_selection_rule.py <teachers> <inits>` | four additional GPT-2 teachers, their MPS fits and selection-rule floors (Table VI), and multi-seed runs of the key two-spin configurations | Sec. IV.E |
| `e5_robust.py` | pulse-area noise | Table V |
| `eo_sector.py`, `e6_sector.py`, `e6_floor.py`, `e3_scale_prep.py`, `run_e6_*.sh` | `S_z`-sector simulation for 24 dots (n=8) | Sec. IV.F |
| `e7_connect.py`, `run_e7.sh`, `e7_table.py` | chain vs. 2-D array vs. round robin vs. random matchings at equal depth (seed 0) | Sec. IV.D, Table IV |
| `make_figs.py` | all figures | |
| `zzswap_search.py`, `best_x.npy`, `best_pat.txt` | earlier 4-spin logical ZZ(θ)·SWAP pulse search (not used in the paper) | |

## Reproducing the headline numbers

```bash
python3 -m venv venv && ./venv/bin/pip install numpy scipy torch transformers
cd numerics
python3 e3e_selection_rule.py            # Table II floors (seconds)
python3 e3d_dist_ceiling.py              # Table II optima (minutes; writes results/*.pt cache, 58 MB)
python3 e3_eo.py st cat_V8L2 neel bubble=24 pi 400   # KL 0.014 at R=2N, 12 dots (about a minute)
python3 e3_eo.py eo cat_V8L2 0 bubble=18 pi 300      # KL 0.011 at R=N, 18 dots (10–40 min)
python3 e8_polar_sector.py 2 24 3        # two-qubit polar factors: F_proc = 1 at R=4N
python3 e8_polar_sector.py 3 36,54,90 6 rand0,gpt2_cproj,prod_2x1,identity   # Appendix A, hours
```

Exact command lines for every sweep are in the `run_*.sh` scripts. The `eo` runs at n=8 need 4–8 GB each
and must be run sequentially (`run_e6_eo.sh`).

GPT-2 weights are downloaded by `transformers` on first use (`gpt2`, ~500 MB).
