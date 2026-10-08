#!/bin/bash
# st n=10 multi-seed on suneo CPU (D=184756): 8 runs in parallel, 4 threads each
cd ~/eoall-numerics; export OMP_NUM_THREADS=4
{
for s in 1 2; do
  echo "st cat_V32L2 neel bubble=60 pi 400 4 $s"
  echo "st begin_V4L5 neel bubble=40 pi 400 4 $s"; echo "st begin_V4L5 neel bubble=60 pi 400 4 $s"
done
echo "st cat_V32L2 alt bubble=60 random 400 4 0"; echo "st begin_V4L5 alt bubble=60 random 400 4 0"
} | xargs -P 8 -L 1 bash -c 'nice -n 10 python3 e10_sector.py $0 "$@"' >> results/e10_suneo.log 2>&1
echo "E10 SUNEO DONE" >> results/e10_suneo.log
