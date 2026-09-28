#!/bin/sh
# E6: scaling to 8 logical qubits (256 outcomes), st n=8 N=16.  Neel (pi init, best at n=6) and alt (random).
export OMP_NUM_THREADS=4
for t in cat_V16L2 begin_V4L4; do
  for R in 16 32 48; do
    python3 e3_eo.py st $t neel bubble=$R pi 400 4
  done
  python3 e3_eo.py st $t alt bubble=32 random 400 4
done
