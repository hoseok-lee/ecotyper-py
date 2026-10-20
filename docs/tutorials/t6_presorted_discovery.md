# Tutorial 6: *De novo* discovery from presorted cell populations

**Goal:** discover cell states and ecotypes from expression profiles that were measured
separately for each cell type, e.g. FACS-sorted populations profiled by bulk RNA-seq.
R equivalent: `EcoTyper_discovery_presorted.R`.

:::{tip}
**In short, with the scverse API** (one AnnData of samples x genes per cell type):

```python
import ecotyper as et
model = et.discovery({"B.cells": adata_b, "Fibroblasts": adata_fib, ...}, n_jobs=32)
```
:::

This is the discovery of [Tutorial 4](t4_bulk_discovery.md) without the CIBERSORTx steps: the
per-cell-type profiles are your measured data rather than CIBERSORTx's purified estimates.

## Inputs

- `ecotyper/example_data/Tutorial_6/PresortedDiscovery/`: one file per cell type
  (`B.cells.txt`, `CD4.T.cells.txt`, … 12 files). Each is a genes x samples matrix with gene
  symbols in the first column, linear scale. **Sample IDs must be the same across files**:
  ecotypes are built from which states co-occur in the same sample.
- `ecotyper/example_data/Tutorial_6/PresortedDiscovery_annotation.txt`: `ID`, `Histology`, …

## Step 1: Load and run

```python
from ecotyper import workflows as wf
from ecotyper.io import read_annotation

profiles = wf.read_profiles("ecotyper/example_data/Tutorial_6/PresortedDiscovery")
ann = read_annotation("ecotyper/example_data/Tutorial_6/PresortedDiscovery_annotation.txt")

res = wf.discover_from_profiles(profiles, ann, scale_column="Histology", n_jobs=32)
```

Expect `Only 6 genes are available for 'NK.cells'; skipping (need >= 50).`: that profile
is too sparse for discovery, and R skips it as well.

## Step 2: Inspect

```python
res.ranks
#            CellType  Chosen_Rank
#           B.cells            8
#       CD4.T.cells            7
#               ...
res.ecotypes.ecotypes.groupby("Ecotype").size()   # states per ecotype (E1..E7)
res.assignment.initial_assignment[["ID", "MaxEcotype", "AssignmentQ", "Ecotype"]]
```

## Step 3: Save the model

```python
wf.write_discovery(res, "DiscoveryOut", name="PresortedDiscovery", fractions="All_genes")
```

Command-line equivalent:

```bash
ecotyper-py discover -i ecotyper/example_data/Tutorial_6/PresortedDiscovery \
  -a ecotyper/example_data/Tutorial_6/PresortedDiscovery_annotation.txt \
  --scale-column Histology -n PresortedDiscovery -o DiscoveryOut -t 32
```

Output layout (identical to R's):

```text
DiscoveryOut/EcoTyper/PresortedDiscovery/All_genes/
├── Analysis/rank_selection/rank_data.txt                 chosen rank per cell type
├── Cell_States/discovery/<cell type>/<rank>/              W.txt, H.txt, state_assignment.txt, gene_info.txt, ...
└── Ecotypes/discovery/ecotypes.txt, ecotype_assignment.txt, jaccard_matrix.txt, ...
```

The parameters are the same as in [Tutorial 4](t4_bulk_discovery.md#parameters).
