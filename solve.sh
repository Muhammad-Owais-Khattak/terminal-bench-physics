#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root
mkdir -p /root/solver
cp /solution/search.py /root/solver/search.py
# smoke test on one development segment
python3 /root/solver/search.py /root/data/segment_00.npz
