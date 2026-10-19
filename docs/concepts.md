# Concepts

## What EcoTyper does

EcoTyper works in two modes:

- **Discovery** finds, for each cell type, the transcriptional **cell states** across samples
  (NMF on per-cell-type expression, with the number of states chosen by the cophenetic
  coefficient). It then groups states that co-occur across samples into **ecotypes** (Jaccard
  similarity of state assignments + hierarchical clustering). The result is a **model**.
- **Recovery** takes a model (prebuilt, or your own discovery result) and assigns new bulk
  samples, single cells or Visium spots to its cell states and ecotypes (NMF with the
  model's fixed W).

The prebuilt models are **Carcinoma** (12 cell types, 10 ecotypes CE1–CE10, discovered across
16 solid tumor types) and **Lymphoma** (DLBCL, 13 cell types, 9 ecotypes LE1–LE9).

## Three API levels

Every workflow in {mod}`ecotyper.workflows` comes in up to three forms:

| Form | Input | Output | Example |
|---|---|---|---|
| AnnData | `AnnData` (samples/cells x genes) | written into `adata.obs`/`obsm`/`uns`, and returned | {func}`~ecotyper.workflows.recover_bulk` |
| `*_frame` | pandas: genes x samples matrix + annotation `DataFrame` | Python result objects | {func}`~ecotyper.workflows.recover_bulk_frame` |
| `*_files` | the R wrappers' input files | the R output folder layout, file for file | {func}`~ecotyper.workflows.recover_bulk_files` |

The `*_files` functions and the [command line](cli.md) are drop-in replacements for the R
scripts. Their outputs can be compared with R's file by file
({mod}`ecotyper.validate`).

## Data orientation

- **pandas level**: like R, matrices are **genes x samples** (rows: gene symbols; columns:
  sample or cell IDs). Annotations are data frames with an `ID` column.
- **AnnData level**: **samples x genes**. `obs` holds the annotation, and `obs_names` are the
  IDs. Convert with {func}`~ecotyper.io.to_anndata` / {func}`~ecotyper.io.from_anndata`.

Expression must be on the **linear scale** (TPM/counts, not log). EcoTyper log-transforms
internally.

## AnnData conventions

With the default `key="ecotyper"`:

| Slot | Contents |
|---|---|
| `obs["ecotyper_state_<cell type>"]` | assigned state (`S01`…), NaN if unassigned |
| `obsm["ecotyper_abundance_<cell type>"]` | state abundances (samples x states) |
| `obs["ecotyper_ecotype"]` | ecotype (`CE1`…) or `Unassigned` (bulk/Visium) |
| `obs["ecotyper_max_ecotype"]`, `obs["ecotyper_ecotype_q"]` | most abundant ecotype and its BH-adjusted p-value |
| `obsm["ecotyper_ecotype_abundance"]` | ecotype abundances (samples x ecotypes) |
| `uns["ecotyper"]` | run parameters, NMF iterations per cell type, full result objects under `"results"` |

## Models

A model is the folder `EcoTyper/<discovery>/<fractions>/` in the R layout (`Analysis/`,
`Cell_States/`, `Ecotypes/`). Recovery functions accept a model name plus
`discovery_root` (the folder that contains `<discovery>/`), a path, or a loaded
{class}`~ecotyper.model.EcoTyperModel`:

```python
from ecotyper.model import EcoTyperModel
m = EcoTyperModel.load("ecotyper/EcoTyper", "Carcinoma")     # Carcinoma/Carcinoma_Fractions
m.cell_type_order                     # ['B.cells', 'PCs', 'CD8.T.cells', ...] (12 cell types)
m.cell_types["Fibroblasts"].n_states  # 12: NMF rank chosen in discovery
m.ecotypes                            # ID, CellType, State, Ecotype: states of each ecotype
```

Discovery writes new models with {func}`~ecotyper.workflows.write_discovery` /
{func}`~ecotyper.workflows.write_discovery_scrna`, which recovery can then load.

## Parallelism and reproducibility

- `n_jobs` (CLI: `-t`) runs cell types and NMF ranks in parallel processes, like R's
  `Number of threads`. Keep BLAS single-threaded (`OMP_NUM_THREADS=1`) for best throughput.
- Discovery is deterministic, seeded exactly as in R (`set.seed(1234 + restart)` per NMF run).
- Recovery takes `seed` (default 42). R's recovery scripts do not seed `NMFpredict`, so R's own
  results vary slightly between runs. Started with `set.seed(42)`, R gives the same output as
  the port.
- `r_exact=True` (default) round-trips intermediate values through R's 15-digit text format,
  where the R pipeline writes a file and reads it back, so ties and ranks come out the same as
  in R.
