#!/bin/bash
cd ~/eoall-numerics; export OMP_NUM_THREADS=4
{
for T in cat_V32L2 begin_V4L5; do
  echo "st $T 0 bubble=60 pi 400 4 0"; echo "st $T 1 bubble=60 pi 400 4 0"
  for s in 1 2; do echo "st $T alt bubble=40 random 400 4 $s"; done
done
} | xargs -P 8 -L 1 bash -c 'nice -n 10 python3 e10_sector.py $0 "$@"' >> results/e10_suneo.log 2>&1
echo "E10 SUNEO2 DONE" >> results/e10_suneo.log
