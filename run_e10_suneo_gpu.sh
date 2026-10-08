#!/bin/bash
# suneo GPU3 slot #37 (15:00-22:00Z): shared V100, 3.5 GB cap, CPU nice 19 / 8 threads
cd ~/eoall-numerics; export CUDA_VISIBLE_DEVICES=3; export EO_GPU_FRAC=0.11; export OMP_NUM_THREADS=8; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
while [ $(date -u +%s) -lt $(date -u -d '2026-10-07 15:01' +%s) ]; do sleep 30; done
nice -n 19 $P e10_sector.py eo cat_V32L2 0 bubble=90 pi 300 8 0 cuda c64
nice -n 19 $P e10_sector.py eo begin_V4L5 0 bubble=60 pi 300 8 0 cuda c64
for T in cat_V64L2 begin_V8L4; do nice -n 19 $P e10_sector.py st $T neel bubble=96 pi 1000 8 0 cuda; done
nice -n 19 $P e10_sector.py eo begin_V4L5 0 bubble=30 pi 300 8 0 cuda c64
echo "E10 SUNEO GPU DONE"
