# Performance: exact mode, fast mode and the alternative solver

ecotyper-py has three ways to run discovery and recovery. By default it reproduces R's results
bit for bit, and two opt-in modes trade that exactness for speed.

| Mode | How to select | Equal to R? | Use it for |
|---|---|---|---|
| **exact** (default) | `exact=True` | same arithmetic as R, file for file (see [Validation](validation.md)) | results you want to compare with, or substitute for, R EcoTyper |
| **fast** | `exact=False`, CLI `--fast` | same algorithm; numbers differ from R by about 1e-10 relative; discrete results identical on all tutorials | large discovery runs when bit-identity with R is not needed |
| **alternative solver** | `nmf_method="cd"`, CLI `--nmf-method cd` (discovery only) | **no**: a different NMF objective | experiments; not EcoTyper-equivalent |

## Exact mode

The default mode emulates R wherever R's floating-point results depend on the order of
operations. That covers the NMF package's C++ update kernels (with or without fused
multiply-adds, matching how R was compiled), x87 long-double sums on x86-64, R's 15-digit number
formatting, and the text round trips between R scripts. Several optimizations keep these
semantics while running faster:

- **Vectorised NMF kernel** (`_nmfk.c`, compiled on first use with the system C compiler). The
  columns of H (and the rows of W) are independent within one Brunet half-step. The kernel
  processes 16–64 of them together, so the compiler can use SIMD instructions without
  reordering any sum. Every element still gets exactly R's sequence of operations, so the
  output is bit-identical to the scalar kernel. This is tested on random inputs, in both FMA
  modes, on Apple silicon and x86-64. The kernel is 2.5–3.6× faster on a Xeon (AVX-512) and
  3–5× faster on Apple silicon. With OpenMP (gcc), the column blocks of one NMF can also be
  split across threads, and the result does not depend on the thread count. Recovery uses
  this to give idle cores to the largest cell types.
- **Number formatting in C** (`rcompat/_fmt.c`): R's `formatReal` for whole matrices, which
  produces the same strings as the Python implementation.
- **Scheduling**: each NMF restart is its own task, and tasks are submitted longest first, so
  the slowest runs do not start last. Rank metrics and the per-cell-type cross-correlations
  of scRNA-seq discovery run in parallel. The Wilcoxon ranks of the cell-type-specific gene
  test are computed once for all cell types, which gives identical p-values.
- **Memory**: workers receive only the slice of the data they need (the cell type's cells
  and the model genes), not the whole expression matrix.

`ECOTYPER_NMF_KERNEL=numba` and `ECOTYPER_FMT=python` switch back to the pure numba/Python
implementations, for example to cross-check results.

## Fast mode (`exact=False`)

```python
model = et.discovery(adata, exact=False, n_jobs=32)
et.recovery(adata_bulk, model, exact=False)
```

Fast mode runs the same algorithm: the same random initialisation (R's seeds), Brunet
multiplicative updates, the eps clamp every 10 iterations, and the same connectivity-based
stopping rule. The updates are computed with BLAS matrix products, though, and the
intermediate values skip R's 15-digit text round trips. Sums are accumulated in a different
order, so W and H differ from R's in the last bits. An NMF run can therefore stop a few
iterations earlier or later, but it converges to the same partition.

Validation against R on x86-64 (all output files of each tutorial):

```{include} _generated/fast_table.md
```

In tutorials 1–4 and 6, all discrete outputs agree with R: chosen ranks, state and ecotype
assignments, filtered genes and states.

Tutorial 5 (scRNA-seq discovery, 11 cell types, up to 9 states) is the exception. The chosen
ranks are the same as R's for every cell type, but for 5 of the 11 cell types the consensus
over NMF restarts lands on a different partition of the cells into initial states (adjusted
Rand index against R 0.70 to 0.97; the other 6 cell types are identical). Downstream files
(state-specific genes, filtered states, ecotypes) then differ for those cell types. NMF has many
local optima, and with 50 restarts of a large matrix the last-bit differences of the BLAS
updates are enough to move a restart to a different optimum. This is a property of the fast
mode, not a bug: the exact mode reproduces all 762 files of tutorial 5 as discrete-equal.
Therefore nothing *guarantees* identical assignments in fast mode on other data, and samples
or cells close to a decision boundary can flip. Use the exact mode when results must match R.

## Alternative NMF solver (`nmf_method="cd"`)

`nmf_method="cd"` replaces EcoTyper's NMF (Kullback–Leibler divergence, Brunet multiplicative
updates) in discovery with **Frobenius-loss NMF by coordinate descent** (scikit-learn's
`NMF(solver="cd")`; install with `pip install "ecotyper[sklearn]"`). It starts from the same
random initialisation and stops by scikit-learn's tolerance rule. The rest of the pipeline is
unchanged: consensus over restarts, cophenetic rank selection, state filters and ecotypes.

This mode is an **extension, not EcoTyper**. The objective is different, so states, ranks and
ecotypes can differ from R's. Agreement with R on the tutorials, as adjusted Rand index (ARI)
of the assignments and chosen ranks:

```{include} _generated/solver_table.md
```

## Memory

Peak memory is reported as the summed PSS (proportional set size) of all processes. Pages
that forked workers share with the parent are counted once. Summed RSS counts them once per
worker, which overstates the real use when the parent holds a large matrix.
`ECOTYPER_MP_CONTEXT=forkserver` (or `spawn`) starts workers without the parent's heap; then
RSS and PSS agree.

## Environment variables

| Variable | Effect |
|---|---|
| `ECOTYPER_NO_FMA=1` | emulate R builds without fused multiply-adds (x86-64 Linux R) |
| `ECOTYPER_LD=0/1` | R's long-double reductions off/on (default: on where long double is 80-bit) |
| `ECOTYPER_NMF_KERNEL=numba` | use the numba NMF kernel instead of the C kernel (same results) |
| `ECOTYPER_FMT=python` | use the Python number formatter (same results) |
| `ECOTYPER_MP_CONTEXT` | worker start method: `fork`, `forkserver` or `spawn` |
| `ECOTYPER_CACHE` | where compiled C kernels are cached (default: the temp directory) |
