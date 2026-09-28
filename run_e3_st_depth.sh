#!/bin/sh
# KL vs depth for st-Neel (cat), pi init (best for Neel); R=12,24 already exist for random, add pi and R=6,18,36
export OMP_NUM_THREADS=4
for R in 6 18 36 24; do
  python3 e3_eo.py st cat_V8L2 neel bubble=$R pi 400 4
done
python3 e3_eo.py st begin_V4L3 neel bubble=36 pi 400 4
python3 e3_eo.py st begin_V4L3 alt bubble=36 random 400 4
# convergence check for Neel (2000 steps)
python3 e3_eo.py st cat_V8L2 neel bubble=24 pi 2000 4
