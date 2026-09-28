#!/bin/bash
# E7 connectivity comparison: 1-D bubble vs 2-D grid vs non-local round-robin vs random matchings.
cd "$(dirname "$0")"
{
for R in 6 12 18 24; do for k in bubble grid roundrobin randmatch; do for m in pi random; do
  echo "st cat_V8L2 neel $k $R $m 400 4"; done; done; done
for R in 16 32; do for k in bubble grid roundrobin randmatch; do
  echo "st cat_V16L2 neel $k $R pi 400 4"; done; done
for k in bubble grid roundrobin randmatch; do
  echo "eo cat_V8L2 0 $k 18 pi 300 4"; echo "eo cat_V8L2 0 $k 9 pi 300 4"; echo "eo cat_V8L2 0 $k 36 pi 300 4"; echo "eo cat_V8L2 0 $k 18 random 300 4"; done
} | xargs -P 6 -L 1 bash -c 'python3 e7_connect.py $0 "$@"' >> results/e7.log 2>&1
echo "E7 DONE" >> results/e7.log
