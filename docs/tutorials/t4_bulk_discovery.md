# Tutorial 4: *De novo* discovery of cell states and ecotypes in bulk data

**Goal:** discover cell states and ecotypes in your own bulk cohort and save them as a model
that the recovery tutorials can use. R equivalent: `EcoTyper_discovery_bulk.R`.

## How discovery works

| Step | R script | Python |
|---|---|---|
| 1. cell type fractions | CIBERSORTx (closed source) | **not ported**: provide precomputed |
| 2. cell-type-specific expression ("HiRes" purification) | CIBERSORTx HiRes | **not ported**: provide precomputed |
| 3. feature extraction: log2, top 1,000 variable genes, scaling (per `scale_column` group) | `state_discovery_extract_features.R` | {func}`~ecotyper.discovery_steps.extract_features` |
| 4. NMF for ranks 2…`max_states`, `restarts` runs each (Brunet, seeded `1234 + restart`) | `state_discovery_NMF.R` | {func}`~ecotyper.nmf.nmf_brunet` |
| 5. rank selection from the cophenetic coefficient (`cophenetic_cutoff`) | `state_discovery_rank_selection.R` | {func}`~ecotyper.discovery_steps.select_ranks` |
| 6. initial states and QC filter | `state_discovery_initial_plots.R`, `..._first_filter.R` | {func}`~ecotyper.discovery_steps.initial_states`, {func}`~ecotyper.discovery_steps.first_filter` |
| 7. ecotypes: Jaccard similarity of state assignments + hierarchical clustering (≥ `min_states` states) | `ecotypes.R` | {func}`~ecotyper.discovery_steps.discover_ecotypes` |
| 8. assign samples to ecotypes | `ecotypes_assign_samples.R` | {func}`~ecotyper.discovery_steps.assign_samples` |

Steps 3–8 are ported and match R output for output (see [Validation](../validation.md)).

## Inputs

- A folder with one **genes x samples** profile per cell type, named `<cell type>.txt`: the
  CIBERSORTx HiRes output (step 2). An optional `classes.txt` (one tab-separated line of cell
  type names) fixes the cell type order. Otherwise files are taken in alphabetical order.
- A sample annotation with an `ID` column (optional), and optionally the column to scale by
  (`Histology` here: LUAD and LUSC samples are scaled separately, so that states reflect
  within-histology variation).

In the R tutorial, CIBERSORTx is run on `bulk_lung_data.txt` with the Carcinoma signature
matrix. The HiRes profiles it produces for those 250 TCGA lung samples ship with the example
data in `example_data/Tutorial_6/PresortedDiscovery/` (12 cell types), so this tutorial starts
from there.

## Step 1: Load the profiles

```python
from ecotyper import workflows as wf
from ecotyper.io import read_annotation

profiles = wf.read_profiles("ecotyper/example_data/Tutorial_6/PresortedDiscovery")
{ct: p.shape for ct, p in profiles.items()}      # genes x 250 samples per cell type
ann = read_annotation("ecotyper/example_data/bulk_lung_annotation.txt")
```

## Step 2: Discover states and ecotypes

```python
res = wf.discover_from_profiles(
    profiles, ann, scale_column="Histology",
    max_states=20, restarts=5, cophenetic_cutoff=0.95, min_states=3,
    n_jobs=32)
```

This is the expensive step: (cell types) × (19 ranks) × (5 restarts) NMF runs. With
`n_jobs=32` it takes about 11 minutes. Cell types with fewer than 50 genes are skipped with a
message, as in R (here `NK.cells`, which has 6 genes).

## Step 3: Inspect the results

```python
res.ranks                                  # CellType, Chosen_Rank per cell type
res.rank_tables["Fibroblasts"]             # cophenetic, dispersion, silhouette per rank
res.states["Fibroblasts"].state_assignment # ID, State: samples assigned to each state
res.states["Fibroblasts"].gene_info        # marker genes per state (fold change, ranks)
res.ecotypes.ecotypes                      # CellType, State, Ecotype (E1, E2, ...)
res.ecotypes.jaccard                       # state x state Jaccard matrix used for clustering
res.assignment.assignment                  # sample -> ecotype
```

## Step 4: Save the model and reuse it for recovery

```python
wf.write_discovery(res, "DiscoveryOut", name="MyDiscovery", fractions="Carcinoma_Fractions")
# -> DiscoveryOut/EcoTyper/MyDiscovery/Carcinoma_Fractions/{Analysis,Cell_States,Ecotypes}/...

# recover the new states/ecotypes in another cohort (Tutorial 1 workflow)
wf.recover_bulk(other_adata, "MyDiscovery", discovery_root="DiscoveryOut/EcoTyper")
```

`fractions` names the model's subfolder. R names it after the CIBERSORTx fractions used in step
1 (`Carcinoma_Fractions` with the Carcinoma signature matrix). The file-based equivalent does
steps 1–4 at once:

```bash
ecotyper-py discover -i ecotyper/example_data/Tutorial_6/PresortedDiscovery \
  -a ecotyper/example_data/bulk_lung_annotation.txt --scale-column Histology \
  -n MyDiscovery --fractions Carcinoma_Fractions -o DiscoveryOut -t 32
```

## Parameters

| Parameter | CLI | Default | Meaning |
|---|---|---|---|
| `max_states` | `--max-states` | 20 | NMF ranks tried: 2 to `max_states` |
| `restarts` | `--restarts` | 5 | NMF runs per rank (merged into a consensus matrix) |
| `cophenetic_cutoff` | `--cophenetic` | 0.95 | rank selection threshold; lower gives fewer states |
| `min_states` | `--min-states` | 3 | minimum number of states in an ecotype |
| `scale_column` | `--scale-column` | none | annotation column whose groups are scaled separately |
| `n_jobs` | `-t` | 1 | parallel processes |
