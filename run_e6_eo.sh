#!/bin/sh
# eo n=8 (N=24) on the sector simulator, run sequentially (each run needs ~4-8 GB for autograd; 4 in parallel hit the OOM killer)
export OMP_NUM_THREADS=6
python3 e6_sector.py eo begin_V4L4 0 bubble=24 pi 400 6
python3 e6_sector.py eo cat_V16L2 0 bubble=48 pi 400 6
python3 e6_sector.py eo cat_V16L2 0 bubble=24 random 400 6
