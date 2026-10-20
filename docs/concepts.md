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

## API levels

| Level | Functions | Input | Output |
|---|---|---|---|
| scverse | {func}`ecotyper.recovery`, {func}`ecotyper.discovery` | `AnnData` (dense or sparse, any layer); data type inferred | results in `adata.obs`/`obsm`/`uns`; discovery returns an {class}`~ecotyper.EcoTyperModel` |
| `*_frame` | {func}`~ecotyper.workflows.recover_bulk_frame`, {func}`~ecotyper.workflows.discover_scrna_frame`, ... | pandas: genes x samples matrix + annotation `DataFrame` | Python result objects with every intermediate table |
| `*_files` | {func}`~ecotyper.workflows.recover_bulk_files`, {func}`~ecotyper.workflows.discover_files`, ... | the R wrappers' input files | the R output folder layout, file for file |

All three run the same engine and give the same numbers. The `*_files` functions and the
[command line](cli.md) are drop-in replacements for the R scripts; their outputs can be
compared with R's file by file ({mod}`ecotyper.validate`). The older AnnData wrappers in
{mod}`ecotyper.workflows` (`recover_bulk`, `recover_scrna`, `discover_scrna`) remain for
compatibility; new code should use {func}`ecotyper.recovery` / {func}`ecotyper.discovery`.

## Data types

{func}`ecotyper.infer_data_type` decides from the object:

| Data type | Signal | Needs |
|---|---|---|
| `"visium"` | `obsm["spatial"]` or `uns["spatial"]` | `obs["array_row"]`, `obs["array_col"]`; CIBERSORTx fractions (`fractions=`) |
| `"scRNA"` | `obs[cell_type_key]` exists (default `"CellType"`) | cell type labels named as in the model; `obs[sample_key]` for ecotypes |
| `"bulk"` | neither | nothing else (an annotation in `obs` is carried into the heatmap tables) |
| `"profiles"` (discovery) | a `dict` of AnnData, one per cell type | the same samples in every object |

Ambiguous objects raise an error that says which `data_type=` to pass: spatial coordinates
together with a cell type column (imaging-based single-cell data), or sparse data with many
observations but no cell type column.

## Data orientation

- **pandas level**: like R, matrices are **genes x samples** (rows: gene symbols; columns:
  sample or cell IDs). Annotations are data frames with an `ID` column.
- **AnnData level**: **samples x genes**. `obs` holds the annotation, and `obs_names` are the
  IDs. Convert with {func}`~ecotyper.io.to_anndata` / {func}`~ecotyper.io.from_anndata`.

Expression must be on the **linear scale** (TPM/counts, not log). EcoTyper log-transforms
internally.

## Result keys

With the default `key_added="ecotyper"`:

| Slot | Data types | Contents |
|---|---|---|
| `obs["ecotyper_state_<cell type>"]` | all | assigned state (`S01`…); scRNA: NaN for cells of other types |
| `obsm["ecotyper_abundance_<cell type>"]` | all | state abundances (observations x states) |
| `obs["ecotyper_ecotype"]` | bulk | ecotype (`CE1`…) or `Unassigned` (not significant) |
| `obs["ecotyper_ecotype"]` | scRNA | ecotype that contains the cell's state |
| `obs["ecotyper_max_ecotype"]`, `obs["ecotyper_ecotype_q"]` | bulk | most abundant ecotype and its BH-adjusted p-value |
| `obsm["ecotyper_ecotype_abundance"]` | bulk, Visium | ecotype abundances (observations x ecotypes) |
| `obsm["ecotyper_state_abundance"]` | Visium | fraction-weighted state abundances, columns `<cell type>_<state>` |
| `obs["ecotyper_malignant_fraction"]` | Visium | fraction of the malignant cell type per spot |
| `uns["ecotyper"]["ecotype_abundance"]`, `["ecotype_assignment"]` | scRNA | ecotypes per sample (samples x ecotypes; assignment table) |
| `uns["ecotyper"]["params"]`, `["niter"]` | all | run parameters, NMF iterations per cell type |

Everything stored is h5ad-serialisable (`adata.write_h5ad` works after a run).

## Models

An {class}`~ecotyper.EcoTyperModel` holds, per cell type, the NMF basis W of the cell states and
their marker genes, plus the ecotype definitions. Get one by:

```python
import ecotyper as et

m = et.EcoTyperModel.load("Carcinoma")          # built-in (downloaded and cached on first use)
m = et.discovery(adata)                         # discovered, in memory
m = et.EcoTyperModel.load("models/crc")         # saved with m.save("models/crc")
m = et.EcoTyperModel.load("ecotyper/EcoTyper", "MyDiscovery", "Carcinoma_Fractions")  # R output folder

m.cell_type_order                     # ['B.cells', 'PCs', 'CD8.T.cells', ...]
m.n_states                            # {'B.cells': 9, ...}: NMF rank chosen in discovery
m.ecotypes                            # ID, CellType, State, Ecotype: states of each ecotype
m.discovery_result                    # discovered models: every intermediate table
```

`save()` writes the R layout `<path>/{Analysis,Cell_States,Ecotypes}/...` with the files recovery
reads, so saved models also work with the R scripts. Built-in models are taken from a pinned
commit of the R repository; set `ECOTYPER_MODELS` to a local `EcoTyper/` folder to use it
instead of downloading, and `ECOTYPER_CACHE` to move the cache (default `~/.cache/ecotyper`).

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
