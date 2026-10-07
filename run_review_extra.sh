#!/bin/bash
# Review 2026-10-01: M5 (extra prompts) and M4/6.1 (multi-seed key configurations)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
python3 e3_teacher.py extra >> results/e3_teacher_extra.log 2>&1
python3 e3_mps.py once_V8L2,sky_V4L3,river_V8L2,door_V4L3 >> results/e3_mps_extra.log 2>&1
{
for T in once_V8L2 sky_V4L3 river_V8L2 door_V4L3; do
  echo "st $T neel bubble=24 pi 400 4 0"; echo "st $T alt bubble=24 random 400 4 0"; echo "eo $T 0 bubble=18 pi 300 4 0"; done
for s in 1 2 3 4; do echo "st cat_V8L2 neel bubble=24 pi 400 4 $s"; echo "st cat_V8L2 neel bubble=36 pi 400 4 $s"; echo "st begin_V4L3 neel bubble=24 random 400 4 $s"; done
for s in 1 2; do echo "eo cat_V8L2 0 bubble=18 pi 300 4 $s"; echo "eo cat_V8L2 0 bubble=18 random 300 4 $s"; done
} | xargs -P 2 -L 1 bash -c 'python3 e3_eo.py $0 "$@"' >> results/e3_review_extra.log 2>&1
echo "REVIEW EXTRA DONE" >> results/e3_review_extra.log
