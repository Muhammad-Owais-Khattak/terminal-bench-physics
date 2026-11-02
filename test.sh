#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
chmod 700 /logs/verifier
echo 0 > /logs/verifier/reward.txt

# 72 invocations of at most 45 s each, well inside the 3600 s verifier limit
if timeout 3420 pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA -p no:cacheprovider; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
