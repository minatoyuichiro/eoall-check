#!/bin/bash
cd ~/eoall-numerics; export EO_GPU_FRAC=0.083; export OMP_NUM_THREADS=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=~/eoall-gpu/bin/python
nice -n 10 $P e10_sector.py eo cat_V32L2 0 bubble=30 pi 300 2 0 cuda c64
nice -n 10 $P e10_sector.py eo cat_V32L2 0 bubble=60 pi 300 2 0 cuda c64
echo "E10 GPU B DONE"
