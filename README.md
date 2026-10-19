# ecotyper (Python port)

A Python/AnnData port of **EcoTyper**
(<https://github.com/digitalcytometry/ecotyper>), the framework for
identifying cell states and multicellular communities (ecotypes) from bulk,
single-cell and spatial transcriptomes, by the Newman lab (Stanford).

The port reproduces the R implementation: on the README tutorials (compared
against R on x86-64/glibc), every discrete output is identical, about 60% of the
files are byte-identical, and all numeric outputs agree to ≤8e-11 relative. See
[VALIDATION.md](VALIDATION.md) for the comparison and
[PORTING_PLAN.md](PORTING_PLAN.md) for the mapping from R to Python.

## Credit and license

EcoTyper was developed by Bogdan A. Luca, Chloé B. Steen, Aaron M. Newman and
colleagues. If you use it, cite:

- Luca/Steen et al., *Atlas of clinically distinct cell states and ecosystems
  across human solid tumors*, Cell 2021, <https://doi.org/10.1016/j.cell.2021.09.014>
- Steen/Luca et al., *The landscape of tumor cell states and ecosystems in
  diffuse large B cell lymphoma*, Cancer Cell 2021,
  <https://doi.org/10.1016/j.ccell.2021.08.011>

This port is a derivative of EcoTyper and is distributed under the same
**Stanford non-commercial license** ([LICENSE.md](LICENSE.md)): free for
non-commercial use only. It is not published anywhere.

CIBERSORTx is closed source and is **not** part of the port. Steps that need it
(cell type fractions and high-resolution purification) take precomputed
CIBERSORTx outputs as inputs.

## Installation

```bash
conda create -n ecotyper-py -c conda-forge python=3.12 numpy scipy pandas anndata numba pyicu pyyaml pytest
conda activate ecotyper-py
pip install -e .          # from this folder
```

PyICU is optional; it is used to sort strings the way R does in a UTF-8 locale
(`ECOTYPER_COLLATION=C` switches to byte order, like R with `LC_COLLATE=C`).

## Usage

### AnnData

Samples or cells are `obs` and genes are `var`. Models (prebuilt or
discovered) are folders in the R layout `EcoTyper/<discovery>/<fractions>/`;
the prebuilt Carcinoma and Lymphoma models ship in the R repository's
`EcoTyper/` folder.

```python
import anndata as ad
from ecotyper import workflows as wf
from ecotyper.io import read_bulk_matrix, read_annotation, to_anndata

expr = read_bulk_matrix("example_data/bulk_lung_data.txt")        # genes x samples, R's parser
ann = read_annotation("example_data/bulk_lung_annotation.txt")
adata = to_anndata(expr, ann)                                       # samples x genes

states, ecotypes = wf.recover_bulk(adata, "Carcinoma", discovery_root="ecotyper/EcoTyper", n_jobs=4)

adata.obs["ecotyper_state_Fibroblasts"]           # S01, S02, ... (NaN = not assigned)
adata.obsm["ecotyper_abundance_Fibroblasts"]      # state abundances
adata.obs["ecotyper_ecotype"]                     # CE1..CE10 or "Unassigned"
adata.obsm["ecotyper_ecotype_abundance"]
```

`recover_scrna(adata, model)` (needs `obs["CellType"]`, and `obs["Sample"]`
for ecotypes) and `recover_visium_frame(...)` work the same way. Discovery:
`discover_from_profiles(profiles, annotation)` (purified CIBERSORTx HiRes or
presorted profiles, one genes x samples table per cell type) and
`discover_scrna(adata)`; `write_discovery(...)` writes a model folder that
the recovery functions can load.

### Command line (mirrors the R wrappers, writes the R file layout)

```bash
ecotyper-py recover-bulk  -m bulk_lung_data.txt -a bulk_lung_annotation.txt -d Carcinoma --models ecotyper/EcoTyper -o out
ecotyper-py recover-scrna -m scRNA_CRC_data.txt -a scRNA_CRC_annotation.txt -d Carcinoma --models ecotyper/EcoTyper -o out
ecotyper-py recover-visium -i VisiumBreast -f visium_fractions_example.txt -n VisiumBreast -d Carcinoma --models ecotyper/EcoTyper -o out
ecotyper-py discover -i Tutorial_6/PresortedDiscovery -a Tutorial_6/PresortedDiscovery_annotation.txt -n PresortedDiscovery -o out -t 8
ecotyper-py discover-scrna -m scRNA_CRC_data.txt -a scRNA_CRC_annotation.txt -n discovery_scRNA_CRC -o out -t 8
```

### Randomness

R's random number generator (Mersenne-Twister with R's seeding, `runif`,
`sample()` with rejection sampling) is reimplemented in
`ecotyper.rcompat.rng.RRandom`. Discovery is deterministic as in R
(`set.seed(1234 + restart)` per NMF run). R's recovery scripts call
`NMF::rnmf()` **without** seeding, so R's own recovery results change slightly
from run to run; the port takes an explicit `seed` (default 42), and an R run
started with `set.seed(42)` (see `validation/r_reference/Rprofile.R`) gives
the same output.

## Layout

| module | contents |
|---|---|
| `ecotyper.rcompat` | R semantics: RNG, `sample`, number parsing/formatting, `hclust`/`cutree`/`dist`/`silhouette`, `cor`/`var`/`mean`, quantiles, BH, t-test, Wilcoxon, collation, `merge`/`order` |
| `ecotyper.nmf` | NMF package 0.28: Brunet updates, seeding, connectivity stopping, restarts merging, cophenetic/silhouette/dispersion, `NMFpredict` |
| `ecotyper.recovery` | state and ecotype recovery (bulk, scRNA, Visium) |
| `ecotyper.discovery`, `ecotyper.discovery_scrna` | state and ecotype discovery |
| `ecotyper.workflows` | AnnData and file-based workflows mirroring the R wrappers |
| `ecotyper.model`, `ecotyper.io`, `ecotyper.rlayout` | model loading, readers, R-format writers |
| `ecotyper.validate` | file-by-file comparison with R outputs |

Plots (heatmaps) are not ported.

## Tests

```bash
pytest -m "not e2e"     # unit parity tests against R fixtures in tests/data
pytest -m e2e           # tutorials vs R reference outputs (needs the R references, see validation/)
```
