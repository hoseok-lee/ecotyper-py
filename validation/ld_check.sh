#!/usr/bin/env bash
# Run on the compute server from the repo root: bit-compare rcompat reductions vs R.
set -e
D=work/ld_check; rm -rf $D
work/dock.sh python validation/ld_check.py gen $D
work/dock_r.sh Rscript validation/ld_check.R $D
work/dock.sh python validation/ld_check.py cmp $D
