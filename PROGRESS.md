# Progress log

## 2026-10-16

- Read the whole R code base (wrappers, `pipeline/*.R`, `pipeline/lib/*.R`) and
  README tutorials; wrote `PORTING_PLAN.md`.
- Environments:
  - `ecotyper-r` (conda-forge/bioconda): R 4.4.3, NMF 0.28 and HiClimR 2.2.1
    installed from CRAN source (no osx-arm64 conda builds), matrixTests 0.2.3.1,
    ComplexHeatmap, data.table, etc. OpenBLAS 0.3.34.
  - `ecotyper-py`: Python 3.12, numpy 2.5, scipy 1.18, pandas 3.0, anndata 0.13,
    numba 0.68, PyICU (ICU 78).
- R reference runs started (`validation/r_reference/run_r_reference.sh`, LANG=en_US.UTF-8,
  `set.seed(42)` at R start-up through `R_PROFILE_USER`).
  - Finding: R's recovery output is not reproducible run to run (unseeded
    `rnmf` in `NMFpredict`): a fresh R run of Tutorial 1 differs from the
    shipped `RecoveryOutput` by up to 1.7e-3 in abundances, discrete outputs identical.
  - Finding: `DiscoveryOutput` (Tutorial 4) and `PresortedDiscoveryOutput`
    (Tutorial 6) text outputs are identical: the Tutorial 6 inputs are the
    CIBERSORTx HiRes outputs of Tutorial 4, so Tutorial 4 can be validated
    downstream of CSx.
- Ported R's RNG (MT19937 + set.seed scrambling + Rejection sampling): identical
  draws to R.
- Ported NMF Brunet updates with explicit FMAs where clang contracts them in
  `NMF.so` (found by disassembly). W, H, number of iterations and KL deviance are
  **bit-identical** to R for 1/2/10/11/25 iterations and full runs (510, 1030,
  1660, 2000 iterations).
- Ported hclust (Fortran, incl. the gfortran FMA in average linkage), hcass2,
  cutree, cophenetic, dist, cluster::silhouette, cov.c correlation, R number
  formatting for write.table (verified on 246k values from R output files).

## PAUSED Fri Oct 16 12:25:18 EDT 2026 (power loss, by parent)
- Worker run cancelled; R reference loop (run_r_reference.sh over t1_lymphoma t2_crc t2_lymphoma t3_visium t6_presorted t4_bulk) was killed mid-run during t4_bulk (EcoTyper_discovery_bulk.R). Logs: /tmp/ecotyper_logs/. On resume: check which tutorial reference outputs are complete and rerun the interrupted/missing ones (t4_bulk at least) before continuing.
