#!/bin/sh
# 2-spin S-T, n=6 (N=12): cheap
for T in cat_V8L2 begin_V4L3; do
  for S in alt 1; do
    for cfg in "bubble=12 random" "bubble=12 zero" "bubble=12 pi" "bubble=24 random" "stair=6 random" "bubble=12 mps4"; do
      set -- $cfg
      python3 e3_eo.py st $T $S $1 $2 400 4
    done
  done
done
