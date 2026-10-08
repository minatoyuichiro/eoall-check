#!/bin/bash
# Table IV protocol completion (Codex finding 2): random starts for the n=8 layouts (seeds 1-4) and eo R=36 alternatives
cd ~/eoall-numerics; export OMP_NUM_THREADS=4
{
for s in 1 2 3 4; do for k in bubble grid roundrobin randmatch; do echo "st cat_V16L2 neel $k 32 random 400 4 $s"; echo "st cat_V16L2 neel $k 16 random 400 4 $s"; done; done
for k in grid roundrobin randmatch; do echo "eo cat_V8L2 0 $k 36 random 300 4 0"; done
} | xargs -P 8 -L 1 bash -c 'nice -n 10 python3 e7_connect.py $0 "$@"' >> results/e7_suneo.log 2>&1
echo "E7 SUNEO3 DONE" >> results/e7_suneo.log
