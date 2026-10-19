# Quickstart

Recover the Carcinoma cell states and ecotypes in the TCGA lung bulk samples that ship with
EcoTyper ([example data](example-data)):

```python
from ecotyper import workflows as wf
from ecotyper.io import read_annotation, read_bulk_matrix, to_anndata

expr = read_bulk_matrix("ecotyper/example_data/bulk_lung_data.txt")      # genes x samples
ann = read_annotation("ecotyper/example_data/bulk_lung_annotation.txt")
adata = to_anndata(expr, ann)                                             # samples x genes

states, ecotypes = wf.recover_bulk(adata, "Carcinoma", discovery_root="ecotyper/EcoTyper", n_jobs=4)
```

Results are stored in `adata`:

```python
adata.obs["ecotyper_state_Fibroblasts"].value_counts()   # S01, S02, ... per sample
adata.obsm["ecotyper_abundance_Fibroblasts"]             # samples x states abundances
adata.obs["ecotyper_ecotype"].value_counts()             # CE1..CE10 or "Unassigned"
adata.obsm["ecotyper_ecotype_abundance"]                 # samples x ecotypes
adata.uns["ecotyper"]                                    # parameters, NMF iterations, full results
```

and returned as Python objects: `states` maps each cell type to a
{class}`~ecotyper.recovery.StateRecovery`, and `ecotypes` is an
{class}`~ecotyper.recovery.EcotypeRecovery`.

To get the same files as `Rscript EcoTyper_recovery_bulk.R` instead, use the file-based
function or the [command line](cli.md):

```python
wf.recover_bulk_files("ecotyper/example_data/bulk_lung_data.txt",
                      "ecotyper/example_data/bulk_lung_annotation.txt",
                      "Carcinoma", "out", discovery_root="ecotyper/EcoTyper", n_jobs=4)
# -> out/RecoveryOutput/bulk_lung_data/<cell type>/state_assignment.txt, ...
```

## Next steps

- [Concepts](concepts.md): the three API levels and the AnnData conventions.
- [Tutorials](tutorials/index.md): every EcoTyper workflow on the bundled data.
