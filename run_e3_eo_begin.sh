#!/bin/sh
# reduced begin_V4L3 set for eo (the full 7-config sweep is too slow under load)
for cfg in "bubble=18 random" "bubble=18 mps4"; do
  set -- $cfg
  python3 e3_eo.py eo begin_V4L3 0 $1 $2 300 6
done
