#!/bin/bash
# E10 on 006 GPU0 (shared, 4 GB cap). Slot #30 ends 15:00Z: the caller wraps this in `timeout`; runs checkpoint every 10 steps and resume.
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2
P=~/eoall-gpu/bin/python
for R in 40 60; do nice -n 10 $P e10_sector.py st cat_V32L2 neel bubble=$R pi 400 2 0 cuda; done
nice -n 10 $P e10_sector.py st cat_V32L2 alt bubble=40 random 400 2 0 cuda
for s in 1 2; do nice -n 10 $P e10_sector.py st cat_V32L2 neel bubble=40 pi 400 2 $s cuda; done
nice -n 10 $P e10_sector.py eo cat_V32L2 0 bubble=30 pi 300 2 0 cuda c64
nice -n 10 $P e10_sector.py eo cat_V32L2 0 bubble=60 pi 300 2 0 cuda c64
echo "E10 GPU A DONE"
