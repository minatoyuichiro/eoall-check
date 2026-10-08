#!/bin/bash
cd ~/eoall-numerics; export OMP_NUM_THREADS=8
{
echo "results/e10_st_cat_V32L2_neel_bubble60_pi.json 20 8"
echo "results/e10_st_begin_V4L5_neel_bubble60_pi.json 20 8"
echo "results/e10_st_cat_V64L2_neel_bubble96_pi.json 20 8"
echo "results/e10_st_begin_V8L4_neel_bubble96_pi.json 20 8"
echo "results/e10_eo_cat_V32L2_0_bubble30_pi_c64.json 10 8"
} | xargs -P 5 -L 1 bash -c 'nice -n 10 python3 e10_noise.py $0 "$@"' >> results/e10_noise.log 2>&1
echo "E10 NOISE DONE" >> results/e10_noise.log
