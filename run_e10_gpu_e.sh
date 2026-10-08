#!/bin/bash
# slot #31 remainder: extra seeds and n=14 at 4N
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
run() { f="results/e10_$1.json"; shift; if [ -s "$f" ]; then echo "skip (exists): $f"; else nice -n 10 $P e10_sector.py "$@"; fi; }
for s in 1 2; do for T in cat_V64L2 begin_V8L4; do run st_${T}_neel_bubble96_pi_seed$s st $T neel bubble=96 pi 400 2 $s cuda; done; done
run eo_cat_V32L2_0_bubble90_pi_seed1_c64   eo cat_V32L2 0 bubble=90 pi 300 2 1 cuda c64
run st_cat_V128L2_neel_bubble112_pi_c64    st cat_V128L2 neel bubble=112 pi 400 2 0 cuda c64
run eo_begin_V4L5_0_bubble60_pi_seed1_c64  eo begin_V4L5 0 bubble=60 pi 300 2 1 cuda c64
echo "E10 GPU E DONE"
