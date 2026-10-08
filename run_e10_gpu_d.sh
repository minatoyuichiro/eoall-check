#!/bin/bash
# slot #31 (22:00Z 10/07 - 14:00Z 10/09)
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
while [ $(date -u +%s) -lt $(date -u -d '2026-10-07 22:02' +%s) ]; do sleep 120; done
nice -n 10 $P e10_sector.py st cat_V128L2 neel bubble=56 pi 400 2 0 cuda c64          # resume if cut
nice -n 10 $P e10_sector.py eo cat_V32L2 0 bubble=90 pi 300 2 0 cuda c64
nice -n 10 $P e10_sector.py st cat_V128L2 neel bubble=84 pi 400 2 0 cuda c64
nice -n 10 $P e10_sector.py eo begin_V4L5 0 bubble=60 pi 300 2 0 cuda c64
for T in cat_V64L2 begin_V8L4; do nice -n 10 $P e10_sector.py st $T neel bubble=96 pi 1000 2 0 cuda; done
echo "E10 GPU D DONE"
