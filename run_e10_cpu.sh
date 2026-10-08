#!/bin/bash
# E10 on CPU: st n=10 (N=20, D=184756). Néel from pi init, alt from random; R=2N,3N.
cd "$(dirname "$0")"; export OMP_NUM_THREADS=4
for T in cat_V32L2 begin_V4L5; do
  for R in 40 60; do nice python3 e10_sector.py st $T neel bubble=$R pi 400 4 0; done
  nice python3 e10_sector.py st $T alt bubble=40 random 400 4 0
done >> results/e10_cpu.log 2>&1
echo "E10 CPU DONE" >> results/e10_cpu.log
