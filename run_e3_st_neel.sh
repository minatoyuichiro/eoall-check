#!/bin/sh
export OMP_NUM_THREADS=4
for T in cat_V8L2 begin_V4L3; do
  for cfg in "bubble=12 random" "bubble=24 random" "stair=6 random" "bubble=12 pi"; do
    set -- $cfg
    python3 e3_eo.py st $T neel $1 $2 400 4
  done
done
