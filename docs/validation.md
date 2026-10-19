# Validation against R

**Reference.** The EcoTyper R pipeline (R 4.4.3, NMF 0.28, HiClimR 2.2.1; `docker/Dockerfile.r`)
on x86-64/glibc without FMA contraction. The port ran on the same machine with
`ECOTYPER_NO_FMA=1`, with the same seed and the same inputs. CIBERSORTx outputs are taken as
inputs, not recomputed.

**Method.** Every output file of every tutorial is compared with R's
(`validation/tutorials.py`, {mod}`ecotyper.validate`):

- *byte-identical*: the files are identical byte for byte;
- *discrete-equal*: all non-floating-point cells (labels, ranks, assignments, gene and state
  sets, row and column names) are identical and the shapes match;
- *max rel. diff*: the largest relative difference over all numeric cells, compared as parsed
  doubles.

**Result.** 2,608 files compared: all discrete-equal, none missing, numeric differences
≤ 8e-11 relative (target 1e-8). The remaining differences are last-digit noise from the
order of floating-point operations in a few reductions (e.g. BLAS vs R's loops). None of them
changes a state, ecotype, rank or gene set. The per-tutorial breakdown and a chart are on the
[Benchmarks](benchmarks.md) page.

```{include} ../VALIDATION.md
:start-after: "ecotype, rank or gene set."
:end-before: "- Benchmarks, `exact=False` fast mode"
```
