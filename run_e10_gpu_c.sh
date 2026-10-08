#!/bin/bash
# batch C: waits for batch B (eo n=10) to finish, then st n=10 (begin) and st n=12 on the GPU
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
while pgrep -f "e10_sector.py eo" > /dev/null; do sleep 60; done
for R in 40 60; do nice -n 10 $P e10_sector.py st begin_V4L5 neel bubble=$R pi 400 2 0 cuda; done
nice -n 10 $P e10_sector.py st begin_V4L5 alt bubble=40 random 400 2 0 cuda
for T in cat_V64L2 begin_V8L4; do
  for R in 48 72; do nice -n 10 $P e10_sector.py st $T neel bubble=$R pi 400 2 0 cuda; done
  nice -n 10 $P e10_sector.py st $T alt bubble=48 random 400 2 0 cuda
done
nice -n 10 $P e10_sector.py st cat_V128L2 neel bubble=56 pi 400 2 0 cuda c64
echo "E10 GPU C DONE"
