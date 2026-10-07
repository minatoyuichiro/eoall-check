#!/bin/bash
# E7 multi-seed robustness (review 2026-10-01, M4): seeds 1-4 on top of the seed-0 runs in results/e7_*.json
cd "$(dirname "$0")"
{
for s in 1 2 3 4; do
  for R in 12 24; do for k in bubble grid roundrobin randmatch; do for m in pi random; do
    echo "st cat_V8L2 neel $k $R $m 400 4 $s"; done; done; done
  for R in 16 32; do for k in bubble grid roundrobin randmatch; do
    echo "st cat_V16L2 neel $k $R pi 400 4 $s"; done; done
done
for s in 1 2; do for k in bubble grid roundrobin randmatch; do for m in pi random; do
  echo "eo cat_V8L2 0 $k 18 $m 300 4 $s"; done; done; done
} | xargs -P 4 -L 1 bash -c 'python3 e7_connect.py $0 "$@"' >> results/e7_seeds.log 2>&1
echo "E7 SEEDS DONE" >> results/e7_seeds.log
