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

**Pending rows.** In the first benchmark run, R's discovery tutorials (4–6) reused NMF results
cached on disk by earlier runs: R skips every restart whose `estim.RData` already exists. So
their R times were not real. Those three R runs are being repeated with an empty cache, and
their rows will be filled in when the runs finish.

## Reading the results

- **Recovery (T1–T3)**: Python is 1.1–4.5× faster than R. The gain is largest on the small bulk
  inputs, where fixed costs dominate: R starts an `Rscript` process per cell type and passes
  results between steps through text files.
- **Memory**: on the larger scRNA-seq and Visium inputs, Python uses 2–3× more peak memory than
  R, because each worker process receives its own copy of the full expression matrix. This is
  a known area for improvement.
- **Exactness costs little**: these timings are in the default R-exact mode (long-double
  reductions, R's number formatting and text round-trips). A faster opt-in mode
  (`exact=False`) is planned.

## Reproduce

On a machine with Docker and both images built:

```bash
python validation/benchmark.py --jobs 32 t1_lung t1_lymphoma t2_lymphoma t2_crc t3_visium t6_presorted t4_bulk t5_scrna
python validation/benchmark.py --table
```

Each run writes `validation/bench/{r,py}_<tutorial>.json` (wall time and peak resident memory
summed over all processes in the container, sampled every 0.5 s). This page is regenerated from
those files and from `validation/reports/*.tsv` at every documentation build.
