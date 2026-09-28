#!/bin/bash
# convergence check: st with 2000 Adam steps (vs 400 in run_e3_st.sh)
export OMP_NUM_THREADS=4
for cfg in "bubble=12 random" "bubble=24 random" "stair=6 random"; do
  python3 e3_eo.py st cat_V8L2 alt $cfg 2000 4
done
python3 e3_eo.py st begin_V4L3 alt bubble=12 random 2000 4
python3 e3_eo.py st begin_V4L3 alt bubble=24 random 2000 4
