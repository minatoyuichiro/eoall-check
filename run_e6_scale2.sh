#!/bin/sh
# E6 follow-up: st n=8 deeper and longer (does the gap to chi=4 close with depth or with training?)
export OMP_NUM_THREADS=4
python3 e3_eo.py st cat_V16L2 neel bubble=64 pi 400 4
python3 e3_eo.py st cat_V16L2 neel bubble=48 pi 2000 4
python3 e3_eo.py st cat_V16L2 neel bubble=96 pi 400 4
python3 e3_eo.py st begin_V4L4 neel bubble=64 pi 400 4
