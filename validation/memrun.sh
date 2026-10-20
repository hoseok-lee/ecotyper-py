#!/usr/bin/env bash
# usage: validation/memrun.sh <ref-or-dir> <tutorial> <label> [jobs]  -> validation/bench/mem/<label>.json
# Runs validation/tutorials.py with the package source from a git worktree of <ref> (data from this checkout).
set -u
cd "$(dirname "$0")/.."
ref=$1; tut=$2; label=$3; jobs=${4:-32}
wt=work/wt/$ref
if [ ! -d "$wt" ]; then git worktree add --detach "$wt" "$ref" >/dev/null 2>&1 || exit 1; fi
mkdir -p validation/bench/mem work/logs
rm -rf "work/memrun/$label"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba \
  -e ECOTYPER_NO_FMA=1 -e ECOTYPER_MP_CONTEXT -v "$PWD:/work" -w /work ecotyper-py:dev \
  bash -c "PYTHONPATH=/work/$wt/src python validation/bench_wrap.py $label validation/bench/mem/$label.json -- python validation/tutorials.py $tut --out work/memrun/$label --jobs $jobs --no-compare" \
  > work/logs/memrun_$label.log 2>&1
tail -1 work/logs/memrun_$label.log
