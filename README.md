# ecotyper-py (Python port of EcoTyper)

A Python/AnnData port of **EcoTyper**
(<https://github.com/digitalcytometry/ecotyper>), the framework for
identifying cell states and multicellular communities (ecotypes) from bulk,
single-cell and spatial transcriptomes, by the Newman lab (Stanford).

The port reproduces the R implementation: on the README tutorials (compared
against R on x86-64/glibc), every discrete output is identical, about 60% of the
files are byte-identical, and all numeric outputs agree to ≤8e-11 relative. See
the [validation report](https://hoseok-lee.github.io/ecotyper-py/validation.html) for the comparison.

## Documentation

<https://hoseok-lee.github.io/ecotyper-py/>. Sphinx docs (installation, API reference, step-by-step tutorials, Python vs R
benchmarks with charts) live in `docs/`. They are built and published by `.github/workflows/docs.yml` on every
push to `main`. To build them locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html   # open docs/_build/html/index.html
```

## Credit and license

EcoTyper was developed by Bogdan A. Luca, Chloé B. Steen, Aaron M. Newman and
colleagues. If you use it, cite:

- Luca/Steen et al., *Atlas of clinically distinct cell states and ecosystems
  across human solid tumors*, Cell 2021, <https://doi.org/10.1016/j.cell.2021.09.014>
- Steen/Luca et al., *The landscape of tumor cell states and ecosystems in
  diffuse large B cell lymphoma*, Cancer Cell 2021,
  <https://doi.org/10.1016/j.ccell.2021.08.011>

This port is a derivative of EcoTyper and is distributed under the same
**Stanford non-commercial license** (`LICENSE.md`, included in the package): free for
non-commercial use only. It is an independent port, not affiliated with or endorsed by
the EcoTyper authors.

CIBERSORTx is closed source and is **not** part of the port. Steps that need it
(cell type fractions and high-resolution purification) take precomputed
CIBERSORTx outputs as inputs.

## Installation

```bash
pip install ecotyper-py           # the import name is `ecotyper`
pip install "ecotyper-py[icu]"    # optional: R-identical string sorting via PyICU
```

A C compiler (`cc`, `gcc` or `clang`) is needed at run time: the exact-parity kernels ship as C source and are
compiled on first use. From a source checkout, or with conda:

```bash
conda create -n ecotyper-py -c conda-forge python=3.12 numpy scipy pandas anndata numba pyicu pyyaml pytest
conda activate ecotyper-py
pip install -e .          # from this folder
```

PyICU is optional; it is used to sort strings the way R does in a UTF-8 locale
(`ECOTYPER_COLLATION=C` switches to byte order, like R with `LC_COLLATE=C`).

## Usage

### AnnData (scverse style)

```python
import ecotyper as et

# recovery: bulk, single-cell or Visium, inferred from the AnnData
et.recovery(adata, "Carcinoma", cell_type_key="cell_type", sample_key="patient")
adata.obs["ecotyper_state_Fibroblasts"]           # S01, S02, ... (NaN = other cell types)
adata.obs["ecotyper_ecotype"]                     # CE1..CE10
adata.obsm["ecotyper_abundance_Fibroblasts"]      # state abundances

# discovery from annotated single cells -> in-memory model
model = et.discovery(adata, cell_type_key="cell_type", sample_key="patient", n_jobs=16)
et.recovery(new_adata, model, cell_type_key="cell_type")
model.save("models/crc")                          # R layout, usable by R EcoTyper too
model = et.EcoTyperModel.load("models/crc")
```

- Data types: `bulk` (samples x genes), `scRNA` (cells x genes with cell type labels in
  `obs`), `visium` (spots, with CIBERSORTx fractions passed as `fractions=`). Inferred
  from `obsm["spatial"]` and the cell type column; override with `data_type=`.
- `X` or `layer=` holds raw-scale expression; sparse matrices are accepted.
- The built-in Carcinoma and Lymphoma models are downloaded (~5 MB each) from a pinned
  commit of the R repository on first use and cached (`ECOTYPER_CACHE`; set
  `ECOTYPER_MODELS` to a local `EcoTyper/` folder to skip the download).
- Bulk discovery takes CIBERSORTx HiRes (or presorted) profiles:
  `et.discovery({"B.cells": adata_b, "Fibroblasts": adata_fib, ...})`.

Results are identical to the file-based workflows below (`tests/test_api.py`).

### pandas and files (mirrors the R wrappers)

`ecotyper.workflows` has `*_frame` functions (genes x samples DataFrames plus annotation,
returning every intermediate table) and `*_files` functions that read the R wrappers'
input files and write the R output layout file for file.

### Command line (mirrors the R wrappers, writes the R file layout)

```bash
ecotyper-py recover-bulk  -m bulk_lung_data.txt -a bulk_lung_annotation.txt -d Carcinoma -o out   # built-in model
ecotyper-py recover-scrna -m scRNA_CRC_data.txt -a scRNA_CRC_annotation.txt -d Carcinoma --models ecotyper/EcoTyper -o out
ecotyper-py recover-visium -i VisiumBreast -f visium_fractions_example.txt -n VisiumBreast -d Carcinoma --models ecotyper/EcoTyper -o out
ecotyper-py discover -i Tutorial_6/PresortedDiscovery -a Tutorial_6/PresortedDiscovery_annotation.txt -n PresortedDiscovery -o out -t 8
ecotyper-py discover-scrna -m scRNA_CRC_data.txt -a scRNA_CRC_annotation.txt -n discovery_scRNA_CRC -o out -t 8
```

### Speed and alternative solver

- `exact=True` (default) reproduces R bit-for-bit where R's arithmetic is deterministic (optimised C/numba kernels
  with R's operation order; typically 3-7x faster than R on the single-cell and presorted tutorials).
- `exact=False` (`--fast` on the CLI) uses BLAS updates and skips R's text round trips. Numerics stay within
  about 1e-10 relative of R, and discrete outputs match on tutorials 1-4 and 6. On tutorial 5, 5 of 11 cell types
  land in a different NMF consensus partition (NMF local optima), so use exact mode when you need R parity.
- `nmf_method="cd"` (discovery only, `pip install ecotyper-py[sklearn]`) swaps the Brunet updates for scikit-learn's
  coordinate-descent NMF with a Frobenius loss. It is not R-equivalent: chosen ranks and states differ from R.
  See the [performance page](https://hoseok-lee.github.io/ecotyper-py/performance.html) for timings and agreement numbers.

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
| `ecotyper.recovery_steps` | state and ecotype recovery (bulk, scRNA, Visium) |
| `ecotyper.discovery_steps`, `ecotyper.discovery_scrna` | state and ecotype discovery |
| `ecotyper.api` | `recovery`, `discovery`, `infer_data_type` (scverse-style AnnData API) |
| `ecotyper.workflows` | pandas and file-based workflows mirroring the R wrappers |
| `ecotyper.model`, `ecotyper.builtin` | `EcoTyperModel` (load/save/from discovery), built-in model download |
| `ecotyper.io`, `ecotyper.rlayout` | readers, R-format writers |
| `ecotyper.validate` | file-by-file comparison with R outputs |

Plots (heatmaps) are not ported.

## Tests

```bash
pytest -m "not e2e"     # unit parity tests against R fixtures in tests/data
pytest -m e2e           # tutorials vs R reference outputs (needs the R references, see validation/)
```
