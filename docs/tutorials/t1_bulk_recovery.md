# Tutorial 1: Recovery of cell states and ecotypes in bulk data

**Goal:** assign bulk tumor samples to the cell states and ecotypes of a prebuilt model.
R equivalent: `EcoTyper_recovery_bulk.R`.

## 1.1 Carcinoma model on TCGA lung samples

### Step 1: Load the data

The input is a **genes x samples** matrix on the linear scale (here TPM), gene symbols in the
first column, plus an optional sample annotation with an `ID` column.

```python
from ecotyper.io import read_annotation, read_bulk_matrix, to_anndata

expr = read_bulk_matrix("ecotyper/example_data/bulk_lung_data.txt")
ann = read_annotation("ecotyper/example_data/bulk_lung_annotation.txt")
expr.shape                        # (19776, 250): genes x samples
ann["Histology"].value_counts()   # LUSC 135, LUAD 115

adata = to_anndata(expr, ann)     # 250 samples x 19776 genes, annotation in adata.obs
```

The readers parse numbers exactly as R's `read.delim` does, and they apply R's `make.names`
to sample names (`TCGA-37-...` becomes `TCGA.37...`), so IDs match R's outputs.

### Step 2: Recover states and ecotypes

```python
from ecotyper import workflows as wf

states, eco = wf.recover_bulk(adata, "Carcinoma", discovery_root="ecotyper/EcoTyper", n_jobs=8)
```

For each of the 12 cell types, this runs NMF with the model's fixed W matrix
(`NMFpredict`), assigns every sample to its most abundant state, and then tests each sample
for enrichment in each ecotype's states.

### Step 3: Inspect the results

```python
adata.obs["ecotyper_state_Fibroblasts"].value_counts()
adata.obsm["ecotyper_abundance_Fibroblasts"].head()     # samples x S01..Snn

adata.obs[["ecotyper_ecotype", "ecotyper_ecotype_q"]].head()
adata.obsm["ecotyper_ecotype_abundance"].head()          # samples x CE1..CE10

# ecotype distribution per histology
import pandas as pd
pd.crosstab(adata.obs["Histology"], adata.obs["ecotyper_ecotype"])
```

As in R, a sample gets its most abundant ecotype only if the enrichment is significant
(BH-adjusted q < 0.25) and the sample is assigned to at least one of that ecotype's states.
Otherwise `ecotyper_ecotype` is `Unassigned`. The full objects are in `states[cell_type]`
({class}`~ecotyper.recovery.StateRecovery`: `abundances`, `assignment`, `H`,
`heatmap_data`…) and `eco` ({class}`~ecotyper.recovery.EcotypeRecovery`).

### Step 4 (optional): R-compatible output files

```python
wf.recover_bulk_files("ecotyper/example_data/bulk_lung_data.txt",
                      "ecotyper/example_data/bulk_lung_annotation.txt",
                      "Carcinoma", "RecoveryOut", discovery_root="ecotyper/EcoTyper", n_jobs=8)
```

or on the command line:

```bash
ecotyper-py recover-bulk -m ecotyper/example_data/bulk_lung_data.txt \
  -a ecotyper/example_data/bulk_lung_annotation.txt -d Carcinoma \
  --models ecotyper/EcoTyper -o RecoveryOut -t 8
```

This writes the same files as R:

```text
RecoveryOut/RecoveryOutput/bulk_lung_data/
├── <cell type>/state_abundances.txt      states x samples
├── <cell type>/state_assignment.txt      ID, State, InitialState
├── <cell type>/heatmap_data.txt, heatmap_top_ann.txt
└── Ecotypes/ecotype_abundance.txt, ecotype_assignment.txt
RecoveryOut/EcoTyper/Carcinoma/Carcinoma_Fractions/{Cell_States,Ecotypes}/recovery/...   intermediate files
```

The R script's heatmap PNG/PDF plots are not ported. `heatmap_data.txt` holds the values to
draw them.

## 1.2 Lymphoma model on DLBCL samples

The same workflow applies with the **Lymphoma** model. The annotation columns `COO` and
`schmitz_labels` are carried through to the outputs, so you can compare ecotypes with the
known subtypes:

```python
expr = read_bulk_matrix("ecotyper/example_data/bulk_lymphoma_data.txt")   # 20379 x 75
ann = read_annotation("ecotyper/example_data/bulk_lymphoma_annotation.txt")
adata = to_anndata(expr, ann)
wf.recover_bulk(adata, "Lymphoma", discovery_root="ecotyper/EcoTyper", n_jobs=8)

pd.crosstab(adata.obs["COO"], adata.obs["ecotyper_ecotype"])               # LE1..LE9
```

```bash
ecotyper-py recover-bulk -m ecotyper/example_data/bulk_lymphoma_data.txt \
  -a ecotyper/example_data/bulk_lymphoma_annotation.txt -d Lymphoma \
  --models ecotyper/EcoTyper -o RecoveryOut -t 8
```

## Notes

- **Seed.** Recovery uses `seed=42` for `NMFpredict`'s random start. R does not seed this step,
  so two R runs differ slightly. The port's result equals R's when R is started with `set.seed(42)`.
- **Missing genes.** As in R, model genes that are missing from your matrix are set to 0
  ({func}`~ecotyper.recovery.align_rows`). Use the same gene symbols as the model, or recovery
  quality drops.
