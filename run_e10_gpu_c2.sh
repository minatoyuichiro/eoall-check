#!/bin/bash
# use the rest of slot #30 for the informative n=12 runs; resume n=14 last
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
for T in cat_V64L2 begin_V8L4; do nice -n 10 $P e10_sector.py st $T neel bubble=72 pi 1000 2 0 cuda; done
nice -n 10 $P e10_sector.py st cat_V64L2 neel bubble=96 pi 400 2 0 cuda
for s in 1 2; do for T in cat_V64L2 begin_V8L4; do nice -n 10 $P e10_sector.py st $T neel bubble=72 pi 400 2 $s cuda; done; done
nice -n 10 $P e10_sector.py st begin_V8L4 neel bubble=96 pi 400 2 0 cuda
nice -n 10 $P e10_sector.py st cat_V128L2 neel bubble=56 pi 400 2 0 cuda c64          # resume
echo "E10 GPU C2 DONE"
