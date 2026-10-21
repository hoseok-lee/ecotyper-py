# Validation against the R implementation

Reference: EcoTyper R pipeline (R 4.4.3, NMF 0.28, HiClimR 2.2.1; `docker/Dockerfile.r`)
run on the compute server (x86-64, glibc, no FMA contraction). Port: `ecotyper-py:dev`
(`docker/Dockerfile`) on the same machine with `ECOTYPER_NO_FMA=1`. Same seed, same inputs;
CIBERSORTx outputs are taken as inputs, not recomputed. Every output file of each tutorial
is compared (`validation/tutorials.py`); per-file results are in
[validation/tables.md](validation/tables.md) (raw: `validation/reports/*.tsv`).

"Discrete-equal" = all non-floating-point cells (labels, ranks, assignments, gene/state
sets, row/column names) are identical and file shapes match. Numeric cells are compared
as parsed doubles.

| Tutorial | Files | Byte-identical | Discrete-equal | Missing | Max rel. diff |
|---|---:|---:|---:|---:|---:|
| T1 lung | 175 | 112 | 175 | 0 | 8.2e-13 |
| T1 lymphoma | 189 | 137 | 189 | 0 | 3.5e-13 |
| T2 CRC | 150 | 134 | 150 | 0 | 9.6e-15 |
| T2 lymphoma | 13 | 10 | 13 | 0 | 9.1e-15 |
| T3 Visium | 103 | 84 | 103 | 0 | 1.3e-14 |
| T4 bulk | 608 | 313 | 608 | 0 | 3.5e-12 |
| T5 scRNA | 762 | 490 | 762 | 0 | 8.0e-11 |
| T6 presorted | 608 | 313 | 608 | 0 | 3.5e-12 |

Result: **2608 files compared, 2608 discrete-equal, 0 missing, numeric differences
<= 8e-11 relative** (target 1e-8). Differences are last-digits noise from
floating-point association order in a few reductions (e.g. BLAS vs R's loops); none changes a state,
ecotype, rank or gene set.

## What was needed for exactness on x86

- R on x86-64 accumulates `sum`, `mean`, `colMeans`, `rowMeans`, `cov`/`cor`/`var`/`scale` in
  80-bit `long double`. `src/ecotyper/rcompat/_ld.c` reimplements these kernels (transcribed from
  R 4.4.3 `summary.c`, `array.c`, `cov.c`); used automatically when `long double` is wider than
  `double` (`ECOTYPER_LD=0/1` overrides). `validation/ld_check.sh`: 12/12 kernels bit-identical to R.
- R's `write.table` number formatting (`scientific()` in `format.c`) uses long double too;
  `ld_scientific` reproduces it. `validation/fmt_check.sh`: 462,063 doubles, 0 differences.
  This matters because R re-reads its own files between steps, so a last-digit difference in text
  propagates into ties and ranks.
- Brunet NMF, RNG (Mersenne Twister + R's sampling), hclust, etc. are bit-exact (unit tests).

## Not exact / caveats

- On arm64 (Apple Silicon) `long double == double`, so the numba paths are used; results are
  discrete-equal to the x86 R reference but not byte-identical, because libm `log2` differs from
  glibc by 1 ulp for a few inputs and FMA contraction differs. The reference platform for exactness
  is x86-64/glibc.
- Mac-R vs x86-R outputs differ from each other for the same reason (compiler/libm); the x86 R run is
  the primary reference.
- `ECOTYPER_NO_FMA=1` is needed on CPUs where numba would otherwise emit FMA, to match R's non-FMA
  conda build.
- CIBERSORTx is not reimplemented and was not tested.
- `exact=False` fast mode and the alternative NMF solver: see docs/performance.md (done; fast mode is discrete-equal except t5, where 5 of 11 cell types land in a different NMF partition).

## Benchmarks (R vs Python, x86-64, 32 jobs)

See docs/benchmarks.md for charts and method. Exact mode, current checkout (C Brunet kernel, vectorised number formatting, per-restart scheduling), shared lab server under varying load (t4 was slower than an earlier run for that reason). Python includes numba cold JIT; peak RSS is summed over all processes (double-counts shared pages), PSS counts them once.

| Tutorial | R wall | Python wall | Speed-up (R/Py) | R peak RSS | Python peak RSS |
|---|---|---|---|---|---|
| t1_lung | 2.4 min | 20 s | 7.41x | 11.46 GB | 3.74 GB (1.38 GB PSS) |
| t1_lymphoma | 89 s | 12 s | 7.12x | 7.88 GB | 3.85 GB (1.41 GB PSS) |
| t2_lymphoma | 48 s | 17 s | 2.79x | 1.23 GB | 3.92 GB (0.51 GB PSS) |
| t2_crc | 3.1 min | 66 s | 2.86x | 28.65 GB | 30.98 GB (5.63 GB PSS) |
| t3_visium | 10.3 min | 2.8 min | 3.73x | 37.09 GB | 19.25 GB (3.82 GB PSS) |
| t6_presorted | 44.7 min | 11.1 min | 4.04x | 8.72 GB | 17.48 GB (3.67 GB PSS) |
| t4_bulk | 18.2 min | 15.2 min | 1.20x | 10.43 GB | 17.48 GB (3.12 GB PSS) |
| t5_scrna | 283.5 min | 240.5 min | 1.18x | 82.52 GB | 380.16 GB (14.97 GB PSS) |
