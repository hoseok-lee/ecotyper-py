#!/usr/bin/env bash
# Regenerate the R reference outputs used by the parity tests.
# Usage: run_r_reference.sh <tutorial> [workdir]
#   tutorial: t1_lung t1_lymphoma t2_crc t2_lymphoma t3_visium t4_bulk t5_scrna t6_presorted
# workdir: R working copy = clone of the EcoTyper repo at 11408e7 with
#          example_data/scRNA_CRC_data.txt extracted (default <project>/work/ecotyper-r-src,
#          gitignored so it survives reboots unlike /tmp).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$1"; W="${2:-$HERE/../../work/ecotyper-r-src}"
export PATH=/opt/miniconda3/envs/ecotyper-r/bin:$PATH
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
# one BLAS/OpenMP thread per process; parallelism comes from the job queue
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export R_PROFILE_USER="$HERE/Rprofile.R" ECOTYPER_REF_SEED=42
cd "$W"
case "$T" in
  t1_lung)     Rscript EcoTyper_recovery_bulk.R -d Carcinoma -m example_data/bulk_lung_data.txt -a example_data/bulk_lung_annotation.txt -c Tissue -o RecoveryOutput ;;
  t1_lymphoma) Rscript EcoTyper_recovery_bulk.R -d Lymphoma -m example_data/bulk_lymphoma_data.txt -a example_data/bulk_lymphoma_annotation.txt -c schmitz_labels,COO -o RecoveryOutput ;;
  t2_crc)      Rscript EcoTyper_recovery_scRNA.R -d Carcinoma -m example_data/scRNA_CRC_data.txt -a example_data/scRNA_CRC_annotation.txt -o RecoveryOutput ;;
  t2_lymphoma) Rscript EcoTyper_recovery_scRNA.R -d Lymphoma -m example_data/scRNA_lymphoma_data.txt -a example_data/scRNA_lymphoma_annotation.txt -o RecoveryOutput -c Tissue ;;
  # not a README tutorial: recovery significance (-z TRUE) on the small lymphoma scRNA data
  z_lymphoma)  Rscript EcoTyper_recovery_scRNA.R -d Lymphoma -m example_data/scRNA_lymphoma_data.txt -a example_data/scRNA_lymphoma_annotation.txt -o RecoveryOutputZ -c Tissue -z TRUE -t 2 ;;
  t3_visium)   Rscript EcoTyper_recovery_visium.R -c "$HERE/config_recovery_visium.yml" ;;
  t4_bulk)
    D=CIBERSORTx/hires/MyDiscovery/Carcinoma_Fractions; mkdir -p "$D"
    for f in example_data/Tutorial_6/PresortedDiscovery/*.txt; do ln -sf "$PWD/$f" "$D/"; done
    (cd example_data/Tutorial_6/PresortedDiscovery && ls *.txt | sed 's/\.txt$//' | paste -sd '\t' -) > "$D/classes.txt"
    Rscript EcoTyper_discovery_bulk.R -c "$HERE/config_discovery_bulk.yml" ;;
  t5_scrna)    Rscript EcoTyper_discovery_scRNA.R -c "$HERE/${CONFIG_T5:-config_discovery_scRNA.yml}" ;;
  t6_presorted) Rscript EcoTyper_discovery_presorted.R -c config_discovery_presorted.yml ;;
  *) echo "unknown tutorial $T"; exit 1 ;;
esac
