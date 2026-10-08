#!/bin/bash
# slot #31 on 006 (22:00Z 10/07 - 14:00Z 10/09): n=14 first; the rest only if not already produced on suneo
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
while [ $(date -u +%s) -lt $(date -u -d '2026-10-07 22:02' +%s) ]; do sleep 120; done
run() { f="results/e10_$1.json"; shift; if [ -s "$f" ]; then echo "skip (exists): $f"; else nice -n 10 $P e10_sector.py "$@"; fi; }
run st_cat_V128L2_neel_bubble56_pi_c64        st cat_V128L2 neel bubble=56 pi 400 2 0 cuda c64     # resume from checkpoint
run st_cat_V128L2_neel_bubble84_pi_c64        st cat_V128L2 neel bubble=84 pi 400 2 0 cuda c64
run eo_cat_V32L2_0_bubble90_pi_c64            eo cat_V32L2 0 bubble=90 pi 300 2 0 cuda c64
run eo_begin_V4L5_0_bubble60_pi_c64           eo begin_V4L5 0 bubble=60 pi 300 2 0 cuda c64
run eo_begin_V4L5_0_bubble30_pi_c64           eo begin_V4L5 0 bubble=30 pi 300 2 0 cuda c64
run st_cat_V64L2_neel_bubble96_pi_s1000       st cat_V64L2 neel bubble=96 pi 1000 2 0 cuda
run st_begin_V8L4_neel_bubble96_pi_s1000      st begin_V8L4 neel bubble=96 pi 1000 2 0 cuda
run eo_cat_V32L2_0_bubble60_pi_seed1_c64      eo cat_V32L2 0 bubble=60 pi 300 2 1 cuda c64
echo "E10 GPU D2 DONE"
