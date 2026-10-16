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

## RESUMED 2026-10-16 13:10 (after power loss)
- R reference status: t1_lung, t1_lymphoma, t2_crc, t2_lymphoma, t3_visium complete
  (text outputs copied to `work/r_ref/`, gitignored). t6_presorted and t4_bulk were killed;
  partial outputs moved aside; chain `t6_presorted → t4_bulk → t5_scrna` restarted
  (`/tmp/ecotyper_chain2.sh`, logs `/tmp/ecotyper_logs/chain2.log`).
- Fixed `format_real`: R's `scientific()` decides the number of significant digits with
  double-precision scaling (`alpha = nearbyint(r / 10^kp)`), not correct rounding; ported
  it exactly (e.g. 0.2527597964472405 is written as 0.25275979644724 by R).
- New modules: `model.py` (load EcoTyper model folders), `recovery.py` (state recovery,
  bulk/scRNA ecotype recovery, spatial states/ecotypes), `rlayout.py` (R file layout),
  `workflows.py` (AnnData + file workflows), `io.py`, `validate.py` (tree comparison).
- Tutorial 1 (lung, Carcinoma): all 120 state-recovery files, combined/ecotype abundances
  and RecoveryOutput files byte-identical to R. Only t-test p-values (assignment_p_vals,
  AssignmentP/Q) differ by ≤1.5e-14 relative (scipy's incomplete beta vs R's TOMS 708).
- 13:40 Tutorials 1 (lymphoma), 2 (CRC, lymphoma) and 3 (Visium) also byte-identical to R
  except t-test p-values (≤1.5e-14 rel). R t6 (presorted) reference finished (27 min);
  t4_bulk running, then t5_scrna. Python t6 discovery running (`work/py_t6.log`).
- Found: R's `cov.c` loop is vectorised by clang: for n ≥ 8 the first 8·⌊n/8⌋ products are
  separate mul + sequential add, the tail uses fmadd. Ported in `_var`/`_cor_cols`
  (`cor()` now bit-exact; used by HiClimR::fastCor in scRNA discovery).
- Found: EcoTyper passes the *name* of the scaling column to `scale_data()`, so bulk/presorted
  discovery scales across all samples (not within groups). Reproduced as is.
- pytest: 24 unit parity tests (RNG, sample, rnmf, Brunet bit-exact, consensus/cophenetic/
  silhouettes, hclust/cutree, dist/cor, write.table formatting, t-test, BH, phyper,
  quantile/median/mad, Wilcoxon) — all pass.
