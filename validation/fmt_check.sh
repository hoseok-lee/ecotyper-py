#!/usr/bin/env bash
# Bit-compare number formatting with R's write.table (run from the repo root; needs docker).
set -e
mkdir -p work/tmp_fmt
work/dock_r.sh Rscript validation/fmt_check.R work/tmp_fmt
work/dock.sh python validation/fmt_check.py work/tmp_fmt
