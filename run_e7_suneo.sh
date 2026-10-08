#!/bin/bash
cd ~/eoall-numerics; export OMP_NUM_THREADS=4
{
echo "eo cat_V8L2 0 bubble 18 pi 300 4 2"
echo "eo cat_V8L2 0 bubble 18 random 300 4 2"
echo "eo cat_V8L2 0 grid 18 pi 300 4 2"
echo "eo cat_V8L2 0 grid 18 random 300 4 2"
echo "eo cat_V8L2 0 roundrobin 18 pi 300 4 2"
echo "eo cat_V8L2 0 roundrobin 18 random 300 4 2"
echo "eo cat_V8L2 0 randmatch 18 pi 300 4 2"
echo "eo cat_V8L2 0 randmatch 18 random 300 4 2"
} | xargs -P 5 -L 1 bash -c 'nice -n 10 python3 e7_connect.py $0 "$@"' >> results/e7_suneo.log 2>&1
echo "E7 SUNEO DONE" >> results/e7_suneo.log
