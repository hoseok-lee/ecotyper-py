# Benchmarks: Python vs R

Both implementations ran every tutorial on the same machine: a shared 112-core Intel Xeon Gold
6258R server, x86-64, Linux. Each run used a fresh Docker container (`docker/Dockerfile` for
Python, `docker/Dockerfile.r` for R 4.4.3 + NMF 0.28) with **32 parallel jobs** (R:
`Number of threads`, Python: `n_jobs`) and one BLAS/OpenMP thread per process. Python timings
include numba's just-in-time compilation (cold cache).

```{raw} html
:file: _generated/benchmark_charts.html
```

## Results as tables

```{include} _generated/benchmark_table.md
```

**Note on R discovery timings.** R skips every NMF restart whose `estim.RData` already exists, so the R discovery runs (tutorials 4–6) were repeated with an emptied cache; the rows above are from those clean runs.

## Reading the results

- **Discovery (T4–T6)**: Python is 1.2× (bulk), 4.0× (presorted) and 1.2× (the large scRNA-seq
  tutorial, 4.7 h in R vs 4.0 h in Python) faster. NMF dominates the large run, and both
  implementations use the same algorithm.
- **Recovery (T1–T3)**: Python is 2.8–7.4× faster than R. The gain is largest on the small bulk
  inputs, where fixed costs dominate: R starts an `Rscript` process per cell type and passes
  results between steps through text files.
- **Memory**: summed RSS double-counts pages shared between forked workers; on t5 Python's
  summed RSS is 380 GB but the real footprint (summed PSS) is 15 GB, versus 83 GB summed RSS for R.
  Starting workers with `ECOTYPER_MP_CONTEXT=forkserver` makes summed RSS equal PSS.
- **Exactness costs little**: these timings are in the default R-exact mode (long-double
  reductions, R's number formatting and text round-trips). The opt-in fast mode
  (`exact=False`) is described on the Performance page.

## Reproduce

On a machine with Docker and both images built:

```bash
python validation/benchmark.py --jobs 32 t1_lung t1_lymphoma t2_lymphoma t2_crc t3_visium t6_presorted t4_bulk t5_scrna
python validation/benchmark.py --table
```

Each run writes `validation/bench/{r,py}_<tutorial>.json` (wall time and peak resident memory
summed over all processes in the container, sampled every 0.5 s). This page is regenerated from
those files and from `validation/reports/*.tsv` at every documentation build.
