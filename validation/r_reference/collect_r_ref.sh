#!/usr/bin/env bash
# Copy the text outputs of an R reference run into the reference layout used by validation/tutorials.py.
# Usage: collect_r_ref.sh [R_WORK] [DEST]   (defaults: work/ecotyper-r-src, work/r_ref_x86)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:-$HERE/../../work/ecotyper-r-src}"; D="${2:-$HERE/../../work/r_ref_x86}"
cp_txt() {  # src dst
  [ -d "$W/$1" ] || { echo "skip (missing): $1"; return 0; }
  mkdir -p "$D/$2"
  rsync -a --include='*/' --include='*.txt' --exclude='*' "$W/$1/" "$D/$2/"
  echo "ok: $1 -> $2 ($(find "$D/$2" -type f | wc -l) files)"
}
for P in Carcinoma Lymphoma; do
  for X in bulk_lung_data bulk_lymphoma_data VisiumBreast; do
    cp_txt "EcoTyper/$P/${P}_Fractions/Cell_States/recovery/$X" "$P/states/$X"
    cp_txt "EcoTyper/$P/${P}_Fractions/Ecotypes/recovery/$X" "$P/ecotypes/$X"
  done
  cp_txt "EcoTyper/$P/${P}_Fractions/Cell_States/recovery/scRNA" "$P/states/scRNA"
  cp_txt "EcoTyper/$P/${P}_Fractions/Ecotypes/recovery/scRNA_CRC_data" "$P/ecotypes/scRNA_CRC_data"
done
for O in RecoveryOutput VisiumOutput PresortedDiscoveryOutput; do cp_txt "$O" "$O"; done
cp_txt EcoTyper/PresortedDiscovery/All_genes PresortedDiscovery/All_genes
cp_txt EcoTyper/MyDiscovery/Carcinoma_Fractions MyDiscovery/Carcinoma_Fractions
cp_txt EcoTyper/discovery_scRNA_CRC/Cell_type_specific_genes discovery_scRNA_CRC/Cell_type_specific_genes
