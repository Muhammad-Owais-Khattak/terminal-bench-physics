#!/usr/bin/env bash
set -euo pipefail
printf '\n[1/3] Harbor\n'
harbor --version
printf '\n[2/3] Docker client and daemon\n'
docker version
printf '\n[3/3] Task files\n'
test -f tasks/physical-sciences/physics/gw-glitch-search/task.toml
test -f tasks/physical-sciences/physics/gw-glitch-search/instruction.md
test -f tasks/physical-sciences/physics/gw-glitch-search/solution/search.py
test -f tasks/physical-sciences/physics/gw-glitch-search/tests/test.sh
printf 'All setup checks passed.\n'
