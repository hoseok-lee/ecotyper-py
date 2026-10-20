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

## RESUMED 2026-10-17 00:45 (after reboot; /tmp wiped)
- R source/working copy re-cloned at 11408e7 into `work/ecotyper-r-src` (gitignored via `/work/`),
  `scRNA_CRC_data.txt` extracted. `run_r_reference.sh`, `tests/conftest.py`,
  `validation/tutorials.py` now default to it (`$ECOTYPER_R_WORK` still overrides).
- `work/r_ref/` intact: T1–T3 and T6 R references present; T4/T5 missing.
- 00:49 R chain `t4_bulk → t5_scrna` started (`work/logs/chain3.sh`, logs `work/logs/r_*.log`).
  Python T6 rerun started (`work/logs/py_t6.log`); previous Python T6 run died with the reboot.

## PLAN ADDED 2026-10-17 01:15 (user request), to do AFTER exact parity is finished
Order: finish T4/T5/T6 parity and VALIDATION.md first. Then:
1. **Benchmark.** Time Python against R per tutorial on the same machine and the same job count, with one BLAS/OpenMP thread per process. Reuse R wall times from work/logs where valid; otherwise rerun. Add a timing table to VALIDATION.md (wall time, peak RSS) and a reproducible script `validation/benchmark.py`.
2. **Fast mode** (`exact=False`; the default stays `exact=True`, R-exact). Use BLAS matmuls for the Brunet updates, threaded restarts and fewer copies, and vectorize the hot paths (consensus, cophenetic, filters). Validate on every tutorial by discrete parity (state and ecotype assignments, ranks, filtered genes/states identical to R), and report numeric deviation and speed-up. Document any dataset where the assignments differ, and explain why (NMF converging to a different local minimum).
3. **Alternative NMF solver** (opt-in, e.g. `nmf_method="hals"`, or sklearn/nimfa-style coordinate descent). Use an existing maintained implementation if one is adequate, under the Existing Solutions Preflight. It's explicitly NOT EcoTyper-equivalent. Report agreement with R's assignments (e.g. ARI) and the speed-up, and document it as an extension in the README.
The port is complete only when exact parity + VALIDATION.md + 1–3 are done.

## REMOTE SERVER ADDED 2026-10-17 01:50 (user decision: option A)
- Server: `ssh hoseoklee@142.1.33.110`, NixOS x86-64 (Xeon Gold 6258R), 112 cores, 754 GB RAM, shared lab machine. Use up to ~100 cores.
  Project clone: `~/workspace/ecotyper-py` (on /mnt/data0; /home is 95% full, so never write there).
  Conda packages don't run on NixOS, so Python runs in Docker: `work/dock.sh <cmd>` (image `ecotyper-py:dev`, `docker/Dockerfile`, same package versions as the Mac env).
  `work/r_ref` was rsynced from the Mac, and `work/ecotyper-r-src` is a fresh clone at 11408e7.
- Option A: **R reference runs stay on the Mac** (R's results depend on the platform's compiler/FMA). Python port runs, benchmarks, fast mode and the alternative solver run on the server.
- Server results: 24/24 unit parity tests pass (RNG, Brunet NMF etc. bit-exact cross-platform).
  T1 lung on the server: 175/175 files discrete-equal, 77 byte-identical, max rel 7.39e-13 (29 s with 16 jobs).
  Cause: `log2p1` calls libm `log2` (Apple libm on the Mac vs glibc on Linux), which differs by 1 ulp for a few inputs (9 cells in B.cells/9 expression_matrix), and that propagates.
  So **byte-exact R parity must be checked by Python runs on the Mac**; server runs are discrete-equal with ≤1e-12 drift.
- New TODO (after the plan items): platform-independent libm. Test whether Apple libm's log2/log/exp are correctly rounded on all inputs the tutorials use (compare to an mpmath/CORE-MATH reference). If they are, use correctly-rounded implementations so Python results are byte-identical on the Mac and on Linux. Record the result either way.
- 01:48 server: `work/run_py_all.sh` started (Python t6 → t4 → t5, 32 jobs each, logs `work/logs/py_*.log` on the server).

## ALL JOBS ON SERVER 2026-10-17 02:00 (user decision; supersedes "R stays on Mac")
- The user wants every compute job on the server. The Mac only orchestrates: it edits code, commits and pushes, and runs no R or Python jobs.
  Mac jobs stopped. The Mac's partial T4 R outputs are kept in the Mac's work/ecotyper-r-src (not deleted).
- R image `ecotyper-r:dev` (`docker/Dockerfile.r`): R 4.4.3, NMF 0.28, HiClimR 2.2.1, same conda package versions as the Mac, en_US.UTF-8 collation.
  Run with `work/dock_r.sh <cmd>`. `ECOTYPER_R_THREADS` overrides "Number of threads" in the configs.
  conda-forge builds packages with `-march=nocona` (no FMA), so R on x86 does **no** FMA contraction.
- **New primary reference = R on the server (x86-64/glibc).** Plan: copy server R outputs into `work/r_ref_x86/` (same layout as `work/r_ref/`) and compare Python-on-server against them with `ECOTYPER_NO_FMA=1`. Both sides use glibc libm there, so byte-exact parity should be reachable. Audit the other Mac-specific code paths (`cov.c` clang vectorisation in `_var`/`_cor_cols`, FMAs in hclust) and add an x86/gcc profile where needed.
  Keep the Mac R references (`work/r_ref/`) as a cross-platform comparison: report Mac-R vs x86-R differences in VALIDATION.md.
- 01:58 server: `work/run_r_all.sh` started (R t4 → t5 → t6 → t1 → t2 → t3, 64 threads; logs `work/logs/r_*.log`; outputs in work/ecotyper-r-src).
  `work/run_py_all.sh` running (Python t6 done in 9:18 wall, 35 MB peak RSS (the main process only); now t4, then t5).
- Python T6 (server, FMA on) vs **Mac** R ref: 608 files, 115 byte-identical, 397 discrete-equal, **209 missing**, max rel inf (one or more zero vs nonzero values).
  Unclear whether this is a port bug or platform drift; Python T6 was never compared before. Investigate against the x86 R reference once `r_t6_presorted` finishes; check the missing files first (likely naming/layout).

## STEP 2026-10-17 11:30 (server state + T6 missing files)
- Server R reference runs all finished (t4 15 min, t5 4 h, t6 42 min, t1/t2 1–3 min each). **t3_visium R failed**: `library(Matrix)` is not installed in `ecotyper-r:dev` (add `r-matrix` to `docker/Dockerfile.r`, rebuild, rerun t3). The `rc=` in run_r_all.log is unreliable (it is `$?` of `$(date)`); check each r_*.log instead.
- Python t6/t4/t5 server runs (FMA on) finished earlier (t4 14 min, t5 4.6 h wall).
- New `validation/r_reference/collect_r_ref.sh` builds `work/r_ref_x86/` (text outputs, same layout as `work/r_ref/`) from the server R run; it is built on the server (t3 Visium states/ecotypes are missing until R t3 is rerun).
- Python T6 vs x86 R: same as vs Mac R: 115 byte-identical, 397 discrete-equal, 209 missing. All 209 were `Cell_States/discovery/<ct>/<k>/used.txt`, a file R writes per rank (`"V1"` plus `"X{i}L"\t{i}` for restarts 1..5) that the port never wrote. Fixed with `_write_used` in `workflows.py` (bulk/presorted discovery, cross_cor, and the chosen-rank dir in scRNA).
- 11:30 server: T6 Python rerun started (`work/rerun_t6.sh`, ~10 min, ECOTYPER_NO_FMA=1, output work/py_t6_presorted_b, log work/logs/py_t6_b.log, ends with a compare against r_ref_x86 into validation/reports/t6_presorted.tsv).
- NEXT: read `tail -3 work/logs/py_t6_b.log` on the server; then investigate remaining non-identical T6 files (max rel inf was likely the zero-vs-nonzero values; look at the biggest n_diff_numeric rows). Then compare T4/T5 with `ECOTYPER_NO_FMA=1 ECOTYPER_R_REF=/work/work/r_ref_x86 work/dock.sh python validation/tutorials.py <t> --out work/py_<t> --compare-only`. Also fix the Dockerfile.r Matrix issue and rerun R t3 (can be done in parallel).

## STEP 2026-10-17 16:15 (env bug found; first_filter_data order fixed)
- **Bug: `docker/dock.sh` did not pass `ECOTYPER_NO_FMA` / `ECOTYPER_R_REF` into the container.** So all earlier server Python runs had FMA ON and every earlier `--compare` used the Mac R ref (`work/r_ref`), not `r_ref_x86`. Fixed (`-e ECOTYPER_NO_FMA -e ECOTYPER_R_REF -e ECOTYPER_R_WORK -e ECOTYPER_COLLATION`); verified inside the container that `R_REF=/work/work/r_ref_x86` and `USE_FMA=False`.
- T6 (FMA on, vs Mac R, after used.txt fix): 608 files, 324 byte-identical, 606 discrete-equal, 0 missing, max rel 2e-11 on the rest. Remaining 2 non-discrete: `first_filter_data.txt` (CD4.T.cells/7, Monocytes.and.Macrophages/10) had a different row order. Cause: R's `merge(tb, agg, all.y=T)` with factor key `State` sorts by factor codes, so unmatched y keys stay after matched rows. Fixed with `r_merge(..., x_factor=True)` (unit test added).
- 16:13 server: fresh runs with NO_FMA and the x86 R ref: `work/run_py_nofma.sh a` (t6 then t4, 32 jobs) and `b` (t5, 64 jobs); logs `work/logs/py_nofma_all.log`, `work/logs/py_<t>_nofma.log`; outputs `work/py_<t>`; reports to `validation/reports/<t>.tsv` (untracked on server).
- Still TODO: Dockerfile.r add `r-matrix`, rebuild, rerun R t3; then T1/T2/T3 Python vs x86 R.
- NEXT: when t6/t4 finish (~25 min) read the summary line at the end of `work/logs/py_t6_presorted_nofma.log`; look at the biggest non-identical files. t5 takes ~4.6 h.

## STEP 2026-10-17 22:55 (NO_FMA server results vs x86 R; root cause = x87 long double)
- Server NO_FMA runs vs `r_ref_x86` (reports in server `validation/reports/*.tsv`, untracked):
  T6 and T4: 608 files, 272 byte-identical, 608 discrete-equal, 0 missing, max rel 3.26e-07 (above the 1e-8 target).
  T5: 762 files, 235 byte-identical, 568 discrete-equal, **33 missing**, max rel inf. All 161 non-discrete T5 files are in `discovery_scRNA_CRC/Cell_type_specific_genes`.
  Missing T5 files: `Cell_States/discovery/<ct>/expression_full_matrix_log2.txt` (11) and `Cell_States/discovery_cross_cor/<ct>/expression_full_matrix_{log2,scaled}.txt` (22). They are probably only written in some code path (check workflows.py scRNA writer against R `EcoTyper_discovery_scRNA.R`).
  The T5 mismatches start at the DE step: `<ct>_cell_type_specific_genes_raw.txt` FC differs by ~1e-14 relative (B.cells BANK1 2.29230937156139 R vs ...141 Py) and genes with tied FC come out in a different order, which then propagates (different gene sets -> W, heatmaps).
- **Root cause (hypothesis, not yet verified by a fix): `rcompat/stats.py` assumes `long double == double` (true on the Mac's arm64).** R on x86-64 accumulates `sum`, `mean`, `colMeans`, `rowMeans`, and the `cov.c` cov/cor/var/scale sums in 80-bit x87 long double. The port is therefore not bit-exact vs x86 R. This also explains the T6/T4 1e-7 drift.
- Plan (next steps):
  1. `docker/Dockerfile`: gcc added (done in this commit). Rebuild `ecotyper-py:dev` on the server.
  2. Add a C kernel library (`src/ecotyper/rcompat/_ld.c`, built on demand with `cc -O2 -ffp-contract=off`, loaded via ctypes) implementing with `long double`: `r_sum`, `r_mean` (two-pass), `colMeans`/`rowMeans`, `mean2`+`var`+`cov/cor` of cov.c (cov_complete1/2, cov_na_1), `scale` (colMeans + sqrt(sum(v^2)/(n-1))). Use it when `sizeof(long double) > 8` (x86); keep the numba double path on arm64 (long double == double). Don't write the kernels from memory: check the R 4.4.3 sources (`src/main/summary.c`, `src/library/stats/src/cov.c`, `src/main/array.c`) for the exact LDOUBLE variable types.
  3. Re-run the T5 `Cell_type_specific_genes` step and T6/T4 on the server and compare. Expect byte-identical or <1e-12.
  4. Then fix the 33 missing T5 files.
- The `docker ps` listing now shows no ecotyper jobs running on the server.

## STEP 2026-10-18 02:10 (x87 long-double kernels written; bit-identical to R on x86)
- New `src/ecotyper/rcompat/_ld.c` + `_ld.py` (ctypes, built on first use with `cc -O2 -ffp-contract=off`, OpenMP off by default via OMP_NUM_THREADS=1). Transcribed from R 4.4.3 `summary.c` (rsum, real_mean), `array.c` (colMeans/rowMeans), `cov.c` (cov_na_1, cov_na_2, MEAN_, COV_SDEV: xm[] rounded to double, sums/divisions in long double, LD sd in cor of two vectors) and scale (`colMeans(na.rm)`, `sum(v^2)`). `stats.py` dispatches to it when `np.finfo(longdouble).nmant > 52` (x86). On arm64 the numba paths are unchanged. Override with `ECOTYPER_LD=0/1` (dock.sh now passes it).
- New `validation/ld_check.{py,R,sh}`: bit-compares sum/mean/colMeans/rowMeans/apply-mean/var/row var/cor/cor2/scale (rows and vec)/col-normalize on a 300x40 ill-conditioned matrix against R in `ecotyper-r:dev`. Server result with the kernels: **all 12 IDENTICAL**. With `ECOTYPER_LD=0`: all 12 DIFF (rel 1e-15..9e-12), so the check does discriminate.
- 02:08 server: reruns started with kernels, NO_FMA, x86 R ref: `work/run_py_nofma.sh a` (t6 then t4, 32 jobs) and `b` (t5, 64 jobs). Logs `work/logs/py_nofma_all.log`, `py_<t>_nofma.log`; reports `validation/reports/<t>.tsv` on the server. t6 ~10 min, t4 ~15 min, t5 ~4.6 h.
- Still unconverted: other numpy `.sum()/.mean()` in discovery_scrna.py/discovery.py that R does with LD (DE fold change in `Cell_type_specific_genes` — check which wrapper computes FC and whether it goes through `r_mean`/`col_means`); cumsum if used.
- NEXT: read the summary line of `py_t6_presorted_nofma.log` and `py_t4_bulk_nofma.log` (expect byte-identical or <1e-12). When t5 finishes check `Cell_type_specific_genes` first; then fix the 33 missing T5 files (`expression_full_matrix_log2/scaled.txt` per cell type in discovery/discovery_cross_cor), rerun R t3 (Dockerfile.r now has r-matrix; rebuild image) and compare T1/T2/T3.

## STEP 2026-10-18 07:10 (long-double results; T5 gaps fixed)
- Server NO_FMA + LD kernels vs x86 R ref: **T6 and T4: 608 files, 608 discrete-equal, 0 missing, max rel 5.5e-12** (289 byte-identical). T5 (2.7 h): 762 files, 727 discrete-equal, 33 missing, 2 non-discrete.
- T5 leftovers: (a) 33 missing = `expression_full_matrix_{log2,scaled}.txt` are only written with `write_large` (tutorials.py never set it) and `CrossCor` lacked the full scaled matrix; added `CrossCor.scaled_full`, written to discovery_cross_cor, and `validation/tutorials.py` now passes `write_large=True`. (b) `Ecotypes/discovery/ecotypes.txt` lacked R's extra `Cell type` column (copy of CellType, R ecotypes_scRNA.R:152); added in `write_discovery_scrna`. (c) `Cell_States/discovery/Fibroblasts/full_FCs.txt`: 8 cells differ only in the last digit (~1e-16 abs; the rel 40.8 is a near-zero value); not yet confirmed as pure ULP noise; the DE values now go through the LD kernels but FC of full_FCs (get_fc_for_clusters) may still use numpy sum/mean; check if easy.
- 07:10 server: T5 rerun to verify (a)/(b).
- NEXT: when the T5 rerun finishes (~2.7 h; log work/logs/py_t5_scrna_nofma.log) read the summary line; then rerun R t3 (Dockerfile.r has r-matrix; rebuild ecotyper-r:dev) and compare T1/T2/T3; then generate VALIDATION.md.

## STEP 2026-10-18 12:05 (T5 result; number-formatting root cause on x86)
- T5 server rerun (NO_FMA + LD kernels) vs x86 R: 762 files, 0 missing, 761 discrete-equal, 324 byte-identical. Only non-discrete file: `Fibroblasts/full_FCs.txt` (8 cells in `Rank`/ties differ by 1.0).
- **Root cause: not arithmetic but text formatting.** R's `scientific()` (format.c, R 4.4.3) uses `long double r_prec`, `tbl[]` (double literals widened to LD), `powl`, `nearbyintl` on x86; the port emulated it in double. Values like log2(202)=7.6582114827517946 sit at the 15-digit boundary: R writes `7.65821148275179`, the port wrote `7.6582114827518`; R then re-reads files, so a 1e-15 diff propagated (ties in MaxFC/ranks). log2 itself is bit-identical to glibc (checked 2M values).
- Fix: `ld_scientific` in `_ld.c` (R's algorithm in long double), used by `rcompat/io._scientific` when the LD kernels are active (x86); the double emulation stays for arm64. Also `-0` now prints as `0` (as R). New `validation/fmt_check.{R,py,sh}`: 462,063 doubles, **0 differences vs R write.table** (212 before tbl[] literals fix; ~hundreds with the double emulation).
- 12:03 server: reruns started with the fix: `work/run_py_nofma.sh a` (t6 then t4, 32 jobs, ~30 min) and `b` (t5, 64 jobs, ~3.3 h); logs `work/logs/py_nofma_all.log`, `py_<t>_nofma.log`; reports in server `validation/reports/*.tsv` (server copies of t4/t5/t6 reports are backed up in `work/reports_bak/`).
- NEXT: read summary lines of `py_t6_presorted_nofma.log`/`py_t4_bulk_nofma.log` (expect more byte-identical files, max rel <1e-8), then t5 when done. Then rebuild `ecotyper-r:dev` (Dockerfile.r now has r-matrix), rerun R t3 and compare T1/T2/T3 (Python vs x86 R), then `validation/make_validation_md.py` -> VALIDATION.md.

## STEP 2026-10-18 17:00 (T4/T5/T6 validated vs x86 R)
- Server NO_FMA + LD kernels + long-double formatting vs x86 R ref (finished 15:43): **T6 608 files, 313 byte-identical, 608 discrete-equal, 0 missing, max rel 3.49e-12; T4 identical numbers; T5 762 files, 490 byte-identical, 762 discrete-equal, 0 missing, max rel 8e-11.** All within the 1e-8 target; discrete outputs fully identical. T4/T5/T6 are DONE.
- 16:59 server: rebuilding `ecotyper-r:dev` from Dockerfile.r (adds r-matrix) then running R `t3_visium` (log `work/logs/r_t3_rebuild.log`, outputs `work/ecotyper-r-src/...` -> r_ref_x86 copy per run_r_reference.sh).
- NEXT: check `r_t3_rebuild.log` tail; confirm R t3 produced VisiumOutput in r_ref_x86; run Python T1/T2/T3 (work/dock.sh with NO_FMA=1; see `validation/tutorials.py`) and compare; then `validation/make_validation_md.py` -> VALIDATION.md.

## STEP 2026-10-18 22:00 (R t3 done on x86; Python T1/T2/T3 started)
- Server R `t3_visium` (rebuilt `ecotyper-r:dev` with r-matrix) finished OK (10 min). Ran `validation/r_reference/collect_r_ref.sh` on the server: `work/r_ref_x86` now has Visium states + `VisiumOutput` (T1/T2 R refs were already there).
- 21:59 server: `work/run_py_t123.sh` (detached, NO_FMA, 32 jobs) runs Python t3_visium, t1_lung, t1_lymphoma, t2_crc, t2_lymphoma sequentially; log `work/logs/py_t123_all.log`, per-tutorial `work/logs/py_<t>_nofma.log`, reports in server `validation/reports/<t>.tsv`.
- NEXT: read `tail -n 3 work/logs/py_t123_all.log` and the summary line of each `py_<t>_nofma.log`; fix any mismatch (T3 is new vs x86 R; LD kernels/formatting may already cover it); then `validation/make_validation_md.py` -> VALIDATION.md (copy server reports back, commit).

## STEP 2026-10-19 03:10 (T1/T2/T3 validated; VALIDATION.md written)
- Server Python T1-T3 (NO_FMA + LD kernels) vs x86 R: t3_visium 103 files, max rel 1.3e-14; t1_lung 175, 8.2e-13; t1_lymphoma 189, 3.5e-13; t2_crc 150, 9.6e-15; t2_lymphoma 13, 9.1e-15. All discrete-equal, 0 missing. Together with T4-T6: **all 6 tutorials validated, 2608 files, max rel 8e-11.**
- Wrote VALIDATION.md (summary table, x86 exactness notes, caveats) and validation/tables.md (generated on the server with make_validation_md.py; copied reports back). README claim of "byte-identical" corrected.
- Not yet run: pytest on the server (e2e), Mac-R vs x86-R cross-platform table (optional).
- NEXT: PLAN items. (1) write `validation/benchmark.py` and time R vs Python per tutorial (R wall times in server work/logs r_*.log; Python in py_*_nofma.log) incl. peak RSS; add table to VALIDATION.md. Then (2) exact=False fast mode, (3) alternative NMF solver.

## STEP 2026-10-19 08:10 (benchmark harness; run started)
- Added `validation/benchmark.py` (R vs Python per tutorial, same machine, same job count, fresh container each, 1 BLAS/OMP thread per process), `validation/bench_wrap.py` (wall time + peak summed RSS of all processes in the container, sampled every 0.5 s), and `tutorials.py --no-compare`. Smoke test py t2_lymphoma: 28 s, 8.8 GB peak. Python timings include numba cold JIT.
- Earlier logged times are NOT comparable (R used 64 threads, Python 32 jobs), so everything is rerun at 32 jobs.
- 08:08 server: detached `python3 validation/benchmark.py --jobs 32 t1_lung t1_lymphoma t2_lymphoma t2_crc t3_visium t6_presorted t4_bulk t5_scrna` (log `work/logs/bench_all.log`, per-run logs `work/logs/bench_{r,py}_<t>.log`, results `validation/bench/{r,py}_<t>.json` on the server). Expected ~3 h for everything but t5, then t5 (R ~8 h + Python ~3.3 h). R reruns overwrite `work/ecotyper-r-src` outputs only; the reference copy `work/r_ref_x86` is untouched.
- Server note: moved untracked t4/t5 reports to `work/reports_bak2/` to allow the pull (they are committed on the Mac).
- NEXT: when `bench_all.log` shows `done py t4_bulk`, run `python3 validation/benchmark.py --table` on the server, copy `validation/bench/*.json` back, add the table to VALIDATION.md (t5 row once it finishes), commit. Then item 2 (fast mode); that can be developed on the Mac but timed on the server only after the benchmark finishes (don't disturb timings: the machine is shared, so note load average).

## STEP 2026-10-19 11:15 (benchmark: R discovery timings invalid, rerun queued)
- Benchmark finished t1-t4, t6 and R t5; Python t5 still running (started ~09:00, expect ~3.3 h).
- Recovery rows valid at 32 jobs: t1_lung R 2.4 min / Py 36 s (4.1x); t1_lymphoma 89 s / 20 s (4.5x); t2_lymphoma 48 s / 27 s (1.75x); t2_crc 3.1 / 2.8 min (1.1x); t3_visium 10.3 / 5.5 min (1.9x). Python peak RSS is much higher on t2_crc (86 vs 29 GB) and t3 (82 vs 37 GB): worth a look in fast mode.
- **R discovery rows (t4 1.9 min, t5 7 min, t6 2.3 min) are invalid**: R skips every NMF restart whose `EcoTyper/<name>/.../estim.RData` exists (log: "Skipping NMF ... already exists"). Fix in benchmark.py: move `work/ecotyper-r-src/EcoTyper/{MyDiscovery,PresortedDiscovery,discovery_scRNA_CRC}` to `work/r_cache_bak/<name>.<ts>` before each R discovery run (commit dfad0f5; the server pull from GitHub hung, so the file was copied with scp; server shows it as M until the next pull).
- Server: `work/bench_r_rerun.sh` (nohup) waits for the main benchmark pid 795718, then runs `benchmark.py --impl r --jobs 32 t6_presorted t4_bulk t5_scrna` (appends to `work/logs/bench_all.log`; ends with "R reruns done"). Expect t6 ~40 min, t4 ~15+ min, t5 several hours.
- NEXT: when `bench_all.log` shows "R reruns done": `python3 validation/benchmark.py --table`, copy `validation/bench/*.json` back, add the table to VALIDATION.md, commit. Meanwhile start item 2 (fast mode) on the Mac; don't time on the server until the reruns finish.

## STEP 2026-10-19 (Sphinx documentation, user request)
- Added `docs/` (Sphinx + furo + MyST): installation, quickstart, concepts, CLI (help text generated from `ecotyper.cli`), API reference (autosummary, all modules), the six tutorials step by step, benchmarks page, validation page. Build: `pip install -e ".[docs]"; sphinx-build -b html docs docs/_build/html` (no warnings).
- Benchmarks page: `docs/_scripts/make_benchmark_page.py` builds interactive HTML/SVG charts (wall-time dumbbell on a log scale, peak memory bars, max relative difference vs R with the 1e-8 target; hover tooltips; light and dark mode) and tables from `validation/bench/*.json` + `validation/reports/*.tsv` at every build. Missing results show as "pending".
- `validation/bench/` now committed with the valid JSONs only (R t4/t5/t6 removed because of the NMF cache issue; py t5 still running). When the reruns land: copy the new JSONs, rebuild the docs, update the speed-up sentence in docs/benchmarks.md ("Reading the results") and remove the "Pending rows" paragraph.
- Docstring fixes: workflows module docstring named non-existent functions; recover_scrna_frame had none; rcompat package docstring added.

## STEP 2026-10-19 15:00 (benchmark: R t4/t6 reruns and Python t5 landed)
- Server `bench_all.log`: R t6 and R t4 reruns done (NMF cache moved aside), Python t5 done. Copied `validation/bench/{r_t4_bulk,r_t6_presorted,py_t5_scrna}.json` back.
- Valid discovery rows at 32 jobs: t4_bulk R 18.2 min (10.4 GB) / Py 10.7 min (14.7 GB) = 1.7x; t6_presorted R 44.7 min (8.7 GB) / Py 10.7 min (14.7 GB) = 4.2x; t5_scrna Py 4.35 h, peak summed RSS 369 GB (summed over processes, so it double-counts shared pages; check before quoting).
- Still running: R t5 rerun (started ~14:25, log `work/logs/bench_r_t5_scrna.log`, expect several hours; ends with "R reruns done" in `work/logs/bench_all.log`).
- NEXT: when "R reruns done" appears: copy `r_t5_scrna.json`, rebuild docs (`sphinx-build -b html docs docs/_build/html`), update the speed-up sentence in docs/benchmarks.md, remove "Pending rows" paragraph, add benchmark table to VALIDATION.md, commit. Meanwhile (Mac, no timing) develop item 2: `exact=False` fast mode.

## PLAN ADDED 2026-10-19 16:45 (user request): OBJECTIVE 2 = seamless scverse API
Objective 1 (exact computational parity with R) is achieved. Objective 2, now the top priority (ahead of fast mode and the alternative solver; finishing the R t5 benchmark row stays a small side task):
make ecotyper-py feel like a native scverse / `scanpy.external` tool (reference: https://scanpy.readthedocs.io/en/stable/external/index.html).
- A user who loads data with `sc.read_10x_mtx`, `sc.read_visium` / squidpy, or an AnnData of bulk samples can call something like `ecotyper.recovery(adata, model=...)` and `ecotyper.discovery(adata, ...)`. Names are not fixed; follow scanpy conventions (`key_added`, `copy=False`, `layer=`, results in `adata.obs` / `obsm` / `uns`).
- **Automatic data-type inference** (bulk vs single-cell vs Visium) with an explicit `data_type=` override. Example signals: `obsm["spatial"]` / `uns["spatial"]` → Visium; per-cell annotation in `obs` with many observations and count-like (often sparse) X → scRNA; otherwise bulk. Raise a clear error when ambiguous.
- Accept sparse `X` and layers directly; users never convert or write files themselves.
- **`discovery()` returns an in-memory model object** (`EcoTyperModel`) instead of writing the R directory layout. Saving is optional: `model.save(path)` / `EcoTyperModel.load(path)` (plus loading the R layout and built-in models such as Carcinoma/Lymphoma by name). The R-layout writers stay available for validation.
- Exactness must be preserved: the new API wraps the existing exact engine. Add tests showing the new API gives identical results to the current file-based workflows on small tutorials (t2_lymphoma, t1_lymphoma, t3 subset).
- Update docs (quickstart and tutorials in scanpy style), README and CLI.

## STEP 2026-10-19 23:10 (benchmark complete)
- R t5 rerun finished (283.5 min, 82.5 GB); copied `validation/bench/r_t5_scrna.json`. Full table: t5_scrna R 283.5 min / Py 261.3 min = 1.09x (Py peak summed RSS 369 GB vs 82.5 GB; memory is a known gap: workers each copy the matrix).
- Added table to VALIDATION.md and R-discovery note + discovery bullet to docs/benchmarks.md (removed "Pending rows"). Docs rebuild not rerun this step (charts regenerate at build time).
- NEXT: Objective 2 (scverse API) per "PLAN ADDED 2026-10-19 16:45": start with design of `ecotyper.recovery/discovery`, data-type inference and `EcoTyperModel`. Then fast mode and alt solver.

## STEP 2026-10-20 (Objective 2: scverse API, first version)
- New `ecotyper.api` exported at top level: `et.recovery(adata, model, ...)`, `et.discovery(data, ...) -> EcoTyperModel`, `et.infer_data_type(adata)`. Scanpy conventions: `key_added`, `copy=False`, `layer=`, `cell_type_key`/`sample_key`, results in `obs`/`obsm`/`uns` (h5ad-serialisable; no result objects in `uns`).
  - Data types: bulk / scRNA / visium (recovery), scRNA / profiles (discovery). Inference: `obsm["spatial"]`/`uns["spatial"]` -> visium; `obs[cell_type_key]` -> scRNA; dict -> profiles; else bulk. Errors on spatial + cell type column, and on sparse >2000 obs without a cell type column. Bulk discovery raises with CIBERSORTx guidance; Visium discovery raises.
  - scRNA recovery densifies only the model genes (exact: same rows `align_rows` picks; non-negativity checked on the full matrix). Visium: `fractions=` DataFrame or obsm key, coordinates from `obs[array_row/array_col]`, spot IDs through `make.names` internally and mapped back to `obs_names`.
- `EcoTyperModel`: `from_discovery` (model tables take R's 15-digit text round trip via a temp folder, so recovery equals the file-based chain), `save(path, heatmap=True)` (R layout, R-compatible), `load("Carcinoma"/"Lymphoma")` built-ins via new `ecotyper.builtin` (pinned commit 11408e7, only the ~5 MB recovery files, cache `~/.cache/ecotyper`, `ECOTYPER_MODELS` for a local folder), `n_states`, `ecotype_names`, `repr`, `discovery_result`. `heatmap_genes` falls back to `gene_info.txt` genes (verified equal to heatmap_data.txt rows in all 37 shipped/test models).
- Renamed `ecotyper.recovery` -> `ecotyper.recovery_steps`, `ecotyper.discovery` -> `ecotyper.discovery_steps` (the modules shadowed the new top-level functions). Docs/README references updated.
- CLI: `--models` optional (default: built-in download).
- `tests/test_api.py` (9 tests + 1 opt-in download test, ~4 min): API vs workflows bit-identical for scRNA (sparse, renamed obs keys), scRNA CRC with per-sample ecotypes (+ h5ad round trip), bulk lymphoma, Visium (fractions via obsm, barcodes with "-1"); profiles discovery (4 cell types, ranks <=4, 2 restarts): model W/mapping/ecotypes equal to write_discovery+load, `save()` files byte-identical, recovery with in-memory model equal; scRNA discovery (3 cell types x 150 cells) equal to discover_scrna_frame. Full non-e2e suite: 35 passed (with ECOTYPER_TEST_DOWNLOAD=1).
- Docs: quickstart and concepts rewritten for the scverse API; scverse tip at the top of tutorials 1/2/3/5/6; api.rst has a scverse API section; fixed the validation.md include marker (broken since the benchmark section was added). `sphinx-build -E` clean.
- NEXT: (1) memory: scRNA/bulk workers pickle the full matrix per cell type (369 GB peak on t5) — pass only the cell type's columns/model genes to workers; (2) optional `ecotyper.pl` helpers (state/ecotype heatmaps) or scanpy plotting recipes; (3) then fast mode (`exact=False`) and the alternative NMF solver.

## STEP 2026-10-20 05:45 (5 AM run: memory, fast mode, alt solver, first optimizations)
- **Memory (step 1, done)**. Workers no longer receive the whole matrix: scRNA recovery slices each cell type's cells + model genes in the parent (df7309f, earlier run); bulk/Visium recovery sends only the cell type's model genes (`align_rows` is idempotent). `bench_wrap.py` now also records summed **PSS** (`/proc/<pid>/smaps_rollup`), which counts fork-shared pages once; summed RSS counts the parent heap once per forked worker. Server, 32 jobs (`validation/memrun.sh <ref> <tut> <label>`, worktrees in work/wt, JSON in validation/bench/mem/ on the server):
  | run | summed RSS | PSS | wall |
  |---|---|---|---|
  | t2_crc before (8a836b9) | 85.8 GB | 16.3 GB | 210 s |
  | t2_crc after (645e1b0) | 30.9 GB | 6.1 GB | 131 s |
  | t2_crc after, forkserver | 6.2 GB | 6.2 GB | 128 s |
  | t3 before | 86.6 GB | 15.9 GB | 371 s |
  | t3 after | 19.4 GB | 4.4 GB | 339 s |
  | t3 after, forkserver | 6.2 GB | 4.9 GB | 353 s |
  | t6 after (fork) | 14.7 GB | 5.6 GB | 831 s (server busy) |
  Outputs byte-identical before/after (diff -r). `ECOTYPER_MP_CONTEXT=fork|forkserver|spawn` selects the start method (default: platform). With forkserver, summed RSS = PSS (R-comparable). The real memory (PSS) is the same with fork, so the default is unchanged.
- Bug fixed: numba's on-disk cache reused FMA-compiled kernels after switching `ECOTYPER_NO_FMA=1` (cache keys ignore the `fma` dependency). NO_FMA builds now use a separate cache dir (`rcompat/fma.py`). Server runs were unaffected (fresh container, cold cache).
- **Fast mode (step 2)**: `exact=False` (API/workflows/`--fast` CLI/`tutorials.py --fast`): BLAS Brunet updates (`nmf._brunet_loop_fast`, same init/stop rule/eps clamp), BLAS KL deviance, no R text round trips (`r_exact=False`). Server vs x86 R (reports `validation/reports/<t>_fast.tsv` on the server): t1_lung 175/175 discrete-equal, max rel 9.4e-13; t1_lymphoma 189/189, 3.5e-13; t2_lymphoma 13/13, 1.1e-13; t2_crc 150/150, 2.3e-13; t3 103/103, 4.8e-13; t6 608/608, 1.2e-10 (289 s vs ~640 s exact). t4/t5 running. tests/test_fast.py.
- **Alternative solver (step 3)**: `nmf_method="cd"` (discovery only): scikit-learn `NMF(solver="cd", beta_loss="frobenius")` from the same `rnmf` init, best restart by Frobenius error; not R-equivalent. scikit-learn 1.9.1 added to the Mac env and the Docker image (rebuilt; the previous image is tagged `ecotyper-py:prev`), `ecotyper[sklearn]` extra. `validation/solver_agreement.py` reports chosen ranks and ARI vs R. Server runs for t4/t6/t5 with `--nmf-method cd` not started yet.
- **Optimizations (step 4), exact (group a)**:
  - Profiles (cProfile, 1 job, server): t1_lung 106 s = 58 s writing text (R number formatting in Python) + 34 s NMFpredict + 15 s numba JIT; t2_crc 339 s = 162 s NMFpredict + 144 s writing + 18 s reading.
  - `_nmfk.c`: the Brunet loop in C, blocked over columns so the compiler vectorises it with the same per-element operation order: **bit-identical** to the numba kernels (random shapes, both FMA modes, Mac and server; `tests/test_nmf.py::test_c_kernel_matches_numba`), 2.5-3.6x faster on the server (BW=64 + 512-bit vectors on AVX-512; BW=16 on Apple silicon, 3-5x). Used by default for exact NMF and NMFpredict; `ECOTYPER_NMF_KERNEL=numba` falls back.
  - `_fmt.c`: `format_real` for whole matrices (both the long-double and the double variant of R's scientific()); identical strings on 450k test values (Mac, both variants) and on the R fmt reference; `write_table`/`write_matrix` use it (~5-10x faster on the Mac, much more on the server where Python called ctypes per value).
  - Discovery scheduling: every NMF restart is its own task (was 5 restarts per rank, sequential), submitted longest first (cost ~ size(V) x rank), and the per-rank metrics run in parallel. Recovery tasks also ordered by cost.
  - Local tests: 41 passed, 1 skipped (non-e2e), 125 s (was 230 s).
- 05:40 server: `work/run_exact.sh` (all 8 tutorials, exact, NO_FMA, x86 R ref; reports work/reports_new/<t>.tsv; logs work/logs/new_<t>.log; times in validation/bench/mem/new_<t>.json; t5 parent profiled to work/prof/t5_parent.prof). `work/run_fast.sh` (fast mode) is at t4, then t5. cProfile of t3 (1 job) still running.
- NEXT: check `work/logs/exact_all.log` (expect discrete-equal everywhere and the same byte-identical counts as validation/reports/*.tsv); fast t4/t5 results; cd-solver runs + ARI; t5 parent profile → parallelise serial parent steps (cross_correlation per cell type, DE); final benchmark when the server is quieter; VALIDATION.md + docs/benchmarks.md + docs page on fast mode and the solver.

## STEP 2026-10-20 12:20 (fast-mode results, performance page)
- Server `work/run_exact.sh` finished: all 8 tutorials exact, x86 R ref, NO_FMA: every file discrete-equal (t5 762/762, t6/t4 608/608, t3 103/103, t2_crc 150/150), max rel 8e-11 (t5). t5 exact 14429 s wall (240 min) vs R 283 min.
- Fast mode finished (`work/run_fast.sh`): t1-t4, t6 all discrete-equal (max rel 1.2e-10). **t5 is NOT**: 690/762 discrete-equal. Ranks identical to R for all 11 cell types, but 5 cell types (B, CD8.T, Dendritic, Epithelial, PCs) land in a different NMF consensus partition (ARI of initial states vs R 0.70-0.97; other 6 identical, ARI 1.0). Not a bug: NMF local optima + last-bit BLAS differences; exact mode is identical. docs/performance.md claim ("all discrete outputs agree") corrected accordingly. fast t5 wall 12285 s (exact 14429 s; t5 is dominated by serial/other steps, little fast-mode gain).
- Copied fast reports (`validation/reports/*_fast.tsv`) and timings (`validation/bench/pyfast_*.json`, `validation/bench/mem/`) from the server; docs/performance.md and make_benchmark_page.py (fast/cd tables, pyfast column) committed. Local docs script runs with /opt/miniconda3/envs/ecotyper-py/bin/python.
- NEXT: (1) cd-solver runs on server (`validation/solver_agreement.py`, t4/t6, then t5) -> reports/<t>_cd_agreement.tsv; (2) copy exact reports from work/reports_new into validation/reports if they differ from committed (byte-identical counts), refresh benchmark JSONs (py_* are pre-optimization) and VALIDATION.md; (3) t5 parent profile (work/prof/t5_parent.prof) to speed t5; (4) final cleanup, README, sphinx build, final summary and disable automation.

## STEP 2026-10-20 14:00 (cd-solver runs launched)
- Pushed 3388597 to origin; server pulled (its untracked `*_fast.tsv` reports moved to work/reports_fast_srv/ to let the pull through).
- Server: `work/run_cd.sh t4_bulk t6_presorted t5_scrna` started detached (nohup; `--nmf-method cd`, NO_FMA, x86 R ref, 32 jobs). Logs work/logs/cd_<t>.log and cd_all.log; reports work/reports_new/<t>_cd.tsv; timings validation/bench/mem/cd_<t>.json (server). Server load was ~17 at start (shared machine).
- NEXT: when cd_all.log shows CD_DONE (t5 takes hours), run `validation/solver_agreement.py` per tutorial against work/r_ref_x86 (inside work/dock.sh, sklearn is in the image), copy `validation/reports/<t>_cd_agreement.tsv` + timings, add the table to docs/performance.md. Then: copy exact reports from work/reports_new if they differ from committed; refresh benchmark JSONs (py_* predate optimizations) and VALIDATION.md; t5 parent profile (work/prof/t5_parent.prof) to speed t5; final cleanup, README, sphinx build, final summary, disable automation.

## STEP 2026-10-20 19:05 (cd-solver t4/t6 done; t5 recursion fix)
- Server `work/run_cd.sh` finished: t4 and t6 cd runs OK (29 s python each; reports work/reports_new/t4_bulk_cd.tsv, t6_presorted_cd.tsv; not R-equivalent by design). **t5 cd crashed**: `RecursionError` in `nmf.reorder_dendrogram_order.leaves` (cd's consensus tree is chain-like, depth > 1000). Fixed with an iterative walk (same leaf order; `tests/test_nmf.py::test_reorder_dendrogram_order_deep_chain`).
- NEXT: server `git pull`, relaunch `work/run_cd.sh t5_scrna` detached; then run `validation/solver_agreement.py` for t4/t6 (and t5 when done), copy `<t>_cd_agreement.tsv` + timings, add the table to docs/performance.md. Then the remaining items from the 12:20 / 14:00 NEXT lists.
- 19:06 server pulled 0039937 (pushed); `work/run_cd.sh t5_scrna` relaunched detached (log work/logs/cd_t5_scrna.log, cd_all5.log; ends with CD_DONE). Server load ~58 (shared).
