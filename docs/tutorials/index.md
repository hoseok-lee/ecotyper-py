# Tutorials

These are the six tutorials of the [EcoTyper README](https://github.com/digitalcytometry/ecotyper#readme),
run with the Python port on the example data bundled with the R repository. Each one gives the
AnnData or pandas route, the file-based route that writes the same files as R, and what to look
at in the results.

**Before you start:** install the package and clone the R repository for its example data and
prebuilt models ([Installation › Example data](example-data)). All paths below are relative to
the folder that contains the `ecotyper/` clone.

| # | Tutorial | Data | Model | Python time* |
|---|---|---|---|---|
| 1 | [Recovery in bulk data](t1_bulk_recovery.md) | TCGA lung (250 samples), DLBCL (75 samples) | Carcinoma, Lymphoma | < 1 min |
| 2 | [Recovery in scRNA-seq data](t2_scrna_recovery.md) | colorectal cancer (13,780 cells), lymphoma (1,000 cells) | Carcinoma, Lymphoma | ~3 min |
| 3 | [Recovery in Visium spatial data](t3_visium_recovery.md) | breast cancer Visium (3,813 spots) | Carcinoma | ~6 min |
| 4 | [Discovery in bulk data](t4_bulk_discovery.md) | CIBERSORTx-purified lung profiles | new model | ~11 min |
| 5 | [Discovery in scRNA-seq data](t5_scrna_discovery.md) | colorectal cancer scRNA-seq | new model | ~3.4 h |
| 6 | [Discovery from presorted profiles](t6_presorted_discovery.md) | presorted per-cell-type profiles | new model | ~11 min |

\* Wall time on a 112-core x86 server with 32 parallel jobs. See [Benchmarks](../benchmarks.md).
With fewer jobs, discovery scales roughly linearly with the number of (cell type, rank) NMF
runs per job.

```{toctree}
:maxdepth: 1
:hidden:

t1_bulk_recovery
t2_scrna_recovery
t3_visium_recovery
t4_bulk_discovery
t5_scrna_discovery
t6_presorted_discovery
```
