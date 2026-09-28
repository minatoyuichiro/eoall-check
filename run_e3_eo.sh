#!/bin/sh
# 3-spin EO, n=6 (N=18)
for T in cat_V8L2 begin_V4L3; do
  for cfg in "bubble=18 random" "bubble=18 pi" "bubble=18 zero" "bubble=18 mps4" "stair=9 random" "bubble=36 random" "bubble=36 mps4"; do
    set -- $cfg
    python3 e3_eo.py eo $T 0 $1 $2 300 6
  done
done
