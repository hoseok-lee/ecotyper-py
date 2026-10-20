# Quickstart

`ecotyper` works on {class}`~anndata.AnnData` objects the way scanpy tools do: pass the
object, results land in `adata.obs` / `adata.obsm` / `adata.uns`.

```python
import ecotyper as et
```

## Recovery with a prebuilt model

{func}`ecotyper.recovery` assigns observations to the cell states and ecotypes of a model.
The data type (bulk, single-cell or Visium) is inferred from the object
({func}`~ecotyper.infer_data_type`); pass `data_type=` to override. The built-in models
`"Carcinoma"` and `"Lymphoma"` are downloaded (about 5 MB each) and cached on first use.

### Single cells

```python
import scanpy as sc

adata = sc.read_h5ad("crc_cells.h5ad")       # cells x genes, raw counts (sparse is fine)
adata.obs["cell_type"]                        # labels named like the model's cell types
adata.obs["patient"]                          # sample of each cell (for ecotypes)

et.recovery(adata, "Carcinoma", cell_type_key="cell_type", sample_key="patient")

adata.obs["ecotyper_state_Fibroblasts"]       # S01, S02, ... for fibroblasts, NaN for other cells
adata.obs["ecotyper_ecotype"]                 # ecotype of each cell's state (CE1..CE10)
adata.uns["ecotyper"]["ecotype_abundance"]    # samples x ecotypes
```

Cell type labels must use the model's names (`et.EcoTyperModel.load("Carcinoma").cell_type_order`:
`B.cells`, `PCs`, `CD4.T.cells`, `CD8.T.cells`, `NK.cells`, `Monocytes.and.Macrophages`,
`Dendritic.cells`, `Mast.cells`, `PMNs`, `Fibroblasts`, `Endothelial.cells`,
`Epithelial.cells`). Cells of other types are ignored.

### Bulk samples

```python
bulk = sc.read_h5ad("tcga_lung.h5ad")         # samples x genes, TPM or normalised arrays
et.recovery(bulk, "Carcinoma", n_jobs=4)

bulk.obs["ecotyper_ecotype"].value_counts()   # CE1..CE10 or "Unassigned"
bulk.obsm["ecotyper_ecotype_abundance"]       # samples x ecotypes
bulk.obsm["ecotyper_abundance_Fibroblasts"]   # samples x fibroblast states
```

### Visium

Visium recovery weights the states by cell type fractions per spot, which come from
CIBERSORTx (not part of this package):

```python
vis = sc.read_visium("VisiumBreast/")                         # obs: array_row, array_col
vis.obsm["csx"] = fractions.loc[vis.obs_names]                # spots x cell types (CIBERSORTx)
et.recovery(vis, "Carcinoma", fractions="csx")

vis.obsm["ecotyper_ecotype_abundance"]                        # spots x ecotypes
sc.pl.spatial(vis, color="ecotyper_state_Fibroblasts")
```

## Discovery

{func}`ecotyper.discovery` finds cell states and ecotypes and returns an
{class}`~ecotyper.EcoTyperModel`, which goes straight into {func}`~ecotyper.recovery`:

```python
model = et.discovery(adata, cell_type_key="cell_type", sample_key="patient", n_jobs=16)
model                                   # EcoTyperModel(name='custom', <n> cell types, <n> states, <n> ecotypes)
adata.obs["ecotyper_state_Fibroblasts"] # discovered state of each cell

et.recovery(other_adata, model, cell_type_key="cell_type")

model.save("models/crc")                # R layout: also usable by the R EcoTyper scripts
model = et.EcoTyperModel.load("models/crc")
```

For bulk discovery, deconvolve the mixtures with CIBERSORTx HiRes and pass the purified
profiles, one AnnData (samples x genes) per cell type:
`et.discovery({"B.cells": b_adata, "Fibroblasts": fib_adata, ...})`. Presorted populations
work the same way.

## Exactness

The API wraps the same engine as the file-based workflows, so results equal R's (see
[Validation](validation.md)). A discovered model takes the same 15-digit text round trip
between discovery and recovery as R's files do, so recovery with an in-memory model equals
recovery with the model written to disk.

To reproduce the R scripts' output folders file for file, use the
[file-based workflows](concepts.md#api-levels) or the [command line](cli.md).

## Next steps

- [Concepts](concepts.md): data types, result keys, models, reproducibility.
- [Tutorials](tutorials/index.md): every EcoTyper workflow on the bundled data.
