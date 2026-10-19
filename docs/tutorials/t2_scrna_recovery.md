# Tutorial 2: Recovery of cell states and ecotypes in scRNA-seq data

**Goal:** assign single cells to the cell states of a prebuilt model and, if cells come with a
sample label, compute per-sample ecotype abundances. R equivalent: `EcoTyper_recovery_scRNA.R`.

## Input requirements

- Expression: **genes x cells**, linear scale (counts, CPM or TPM; not log), gene symbols in the
  first column. The file is read with `data.table::fread` semantics
  ({func}`~ecotyper.io.read_scrna_matrix`): cell IDs are kept verbatim, and rows with NA or
  duplicated gene names are dropped.
- Annotation: `ID` (matching the matrix columns), **`CellType`** (one of the model's cell types,
  e.g. `Fibroblasts`, `CD8.T.cells`), and optionally **`Sample`** (enables ecotype recovery).
  Cells of other types are ignored.

## 2.1 Carcinoma model on colorectal cancer cells

### Step 1: Load the data

```python
from ecotyper.io import read_annotation, read_scrna_matrix, to_anndata

expr = read_scrna_matrix("ecotyper/example_data/scRNA_CRC_data.txt")   # 19751 genes x 13780 cells
ann = read_annotation("ecotyper/example_data/scRNA_CRC_annotation.txt")
ann["CellType"].value_counts()      # up to 1,500 cells per cell type
ann["Sample"].nunique()             # 33 samples (tumor and normal)

adata = to_anndata(expr, ann)       # 13780 cells x 19751 genes
```

If you start from your own AnnData object, put the raw-scale matrix in `X` or a layer (pass
`layer=`), and make sure `adata.obs` has `CellType` and `Sample`.

### Step 2: Recover states

```python
from ecotyper import workflows as wf

states, eco = wf.recover_scrna(adata, "Carcinoma", discovery_root="ecotyper/EcoTyper", n_jobs=8)
```

For each cell type, the cells of that type are log2(x+1) transformed, restricted to the model
genes and scaled. NMF with the model's W then gives every cell a state abundance profile, and
the cell is assigned to its top state.

### Step 3: Inspect the results

```python
adata.obs["ecotyper_state_Fibroblasts"].value_counts(dropna=False)   # NaN for non-fibroblasts
adata.obsm["ecotyper_abundance_Fibroblasts"]

# ecotypes are per sample, not per cell: they are returned, not stored in adata
eco.ecotype_abundance          # ecotypes x samples
eco.initial_assignment         # every sample: ID, MaxEcotype, AssignmentP/Q, Ecotype
eco.assignment                 # only samples assigned to an ecotype
```

For each state, the per-sample value is the fraction of that sample's cells of the state's cell
type that are in the state. A sample's ecotype abundance is the mean of these fractions over
the ecotype's states, normalized to sum to 1 across ecotypes.

### Step 4 (optional): R-compatible output files

```bash
ecotyper-py recover-scrna -m ecotyper/example_data/scRNA_CRC_data.txt \
  -a ecotyper/example_data/scRNA_CRC_annotation.txt -d Carcinoma \
  --models ecotyper/EcoTyper -o RecoveryOut -t 8
```

```text
RecoveryOut/RecoveryOutput/scRNA_CRC_data/
├── <cell type>/state_assignment.txt, state_abundances.txt, heatmap_data.txt, heatmap_top_ann.txt
└── Ecotypes/ecotype_abundance.txt, ecotype_assignment.txt
```

Use `-s N` (`subsample=N`) to cap the number of cells per cell type for very large datasets
(sampled with R's RNG, seed 1234, as in R; values below 50 keep all cells).

## 2.2 Lymphoma model on lymphoma cells

The lymphoma example has no `Sample` column, so only cell states are recovered:

```python
expr = read_scrna_matrix("ecotyper/example_data/scRNA_lymphoma_data.txt")   # 14643 x 1000
ann = read_annotation("ecotyper/example_data/scRNA_lymphoma_annotation.txt")  # ID, CellType, Tissue
adata = to_anndata(expr, ann)
states, eco = wf.recover_scrna(adata, "Lymphoma", discovery_root="ecotyper/EcoTyper")
eco is None        # True: no Sample column
```

```bash
ecotyper-py recover-scrna -m ecotyper/example_data/scRNA_lymphoma_data.txt \
  -a ecotyper/example_data/scRNA_lymphoma_annotation.txt -d Lymphoma \
  --models ecotyper/EcoTyper -o RecoveryOut
```

The R tutorial passes `-c Tissue` to add the `Tissue` column to its heatmaps. The port does not
draw heatmaps, and all annotation columns are kept in `adata.obs` anyway.
