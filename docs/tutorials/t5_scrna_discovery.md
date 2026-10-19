# Tutorial 5: *De novo* discovery of cell states and ecotypes in scRNA-seq data

**Goal:** discover cell states and ecotypes directly from annotated single cells.
R equivalent: `EcoTyper_discovery_scRNA.R`. No CIBERSORTx is involved, so the whole pipeline
runs in Python.

## How scRNA-seq discovery works

1. **Cell-type-specific genes** (`filter_genes="cell type specific"`): genes significantly
   higher in a cell type than in all others ({func}`~ecotyper.discovery_scrna.cell_type_specific_genes`).
   This keeps states from being driven by contamination.
2. **Cell-cell correlation.** Per cell type, the top variable genes are scaled and the cell x cell
   correlation matrix is computed ({func}`~ecotyper.discovery_scrna.cross_correlation`).
3. **NMF on the correlation matrix** for ranks 2…`max_states` and rank selection by cophenetic
   coefficient (`cophenetic_cutoff`, 0.975 in this tutorial).
4. **Marker genes and re-discovery.** Marker genes of the initial clusters are extracted, and NMF
   is re-run on expression at the chosen rank
   ({func}`~ecotyper.discovery_scrna.extract_features_scrna`).
5. **QC filter** of states, then **ecotypes** from state co-occurrence across samples (`Sample`
   column), and assignment of samples to ecotypes
   ({func}`~ecotyper.discovery_scrna.discover_ecotypes_scrna`).

## Inputs

Same as [Tutorial 2](t2_scrna_recovery.md): a genes x cells matrix on the linear scale, and an
annotation with `ID`, `CellType` and **`Sample`** (required here, because ecotypes are defined
across samples).

```python
from ecotyper.io import read_annotation, read_scrna_matrix

expr = read_scrna_matrix("ecotyper/example_data/scRNA_CRC_data.txt")   # 19751 genes x 13780 cells
ann = read_annotation("ecotyper/example_data/scRNA_CRC_annotation.txt")
ann.groupby("CellType").size()     # 11 cell types, up to 1,500 cells each
```

## Step 1: Run discovery

```python
from ecotyper import workflows as wf

res = wf.discover_scrna_frame(expr, ann, cophenetic_cutoff=0.975, n_jobs=32)
```

or on AnnData (results also written to `adata.obs["ecotyper_state_<cell type>"]`):

```python
from ecotyper.io import to_anndata
adata = to_anndata(expr, ann)
res = wf.discover_scrna(adata, cophenetic_cutoff=0.975, n_jobs=32)
```

This is the longest tutorial: about 3.4 h with 32 jobs on a 112-core server (R takes
several hours as well). The NMF runs on cell x cell matrices of up to 1,500 x 1,500 per cell
type dominate the run time. Peak memory scales with `n_jobs`.

## Step 2: Inspect the results

```python
res.ranks                                       # chosen number of states per cell type
res.specific["Fibroblasts"].filtered            # cell-type-specific genes used
res.states["Fibroblasts"].state_assignment      # cell -> state
res.states["Fibroblasts"].gene_info             # marker genes per state
res.ecotypes.ecotypes                           # CellType, State, Ecotype
res.assignment.assignment                       # sample -> ecotype
```

## Step 3: Save the model

```python
wf.write_discovery_scrna(res, "DiscoveryOut", "discovery_scRNA_CRC")
# -> DiscoveryOut/EcoTyper/discovery_scRNA_CRC/Cell_type_specific_genes/...
```

The model can then be used for recovery in bulk ([Tutorial 1](t1_bulk_recovery.md)), scRNA-seq
([Tutorial 2](t2_scrna_recovery.md)) or Visium ([Tutorial 3](t3_visium_recovery.md)) data with
`model="discovery_scRNA_CRC"`, `discovery_root="DiscoveryOut/EcoTyper"`. Pass
`write_large=True` to also write the full log2/scaled expression matrices (several GB, needed
only for a file-by-file comparison with R).

Command-line equivalent:

```bash
ecotyper-py discover-scrna -m ecotyper/example_data/scRNA_CRC_data.txt \
  -a ecotyper/example_data/scRNA_CRC_annotation.txt -n discovery_scRNA_CRC \
  --cophenetic 0.975 -o DiscoveryOut -t 32
```

## Parameters

| Parameter | CLI | Default | Meaning |
|---|---|---|---|
| `filter_genes` | `--filter-genes` | `"cell type specific"` | `"cell type specific"`, `"no filter"`, or a number N for the top N variable genes |
| `max_states` | `--max-states` | 20 | NMF ranks tried: 2 to `max_states` |
| `restarts` | `--restarts` | 5 | NMF runs per rank |
| `cophenetic_cutoff` | `--cophenetic` | 0.975 | rank selection threshold |
| `p_value_cutoff` | `--pvalue` | 1.0 | p-value threshold for state enrichment in ecotypes |
| `min_states` | `--min-states` | 3 | minimum number of states in an ecotype |
| `n_jobs` | `-t` | 1 | parallel processes |
