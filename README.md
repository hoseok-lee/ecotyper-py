# Disclaimer

This is an *unofficial* Python implementation for the R package
[EcoTyper](https://github.com/digitalcytometry/ecotyper/), for discovery of
ecotypes in scRNA-seq data, and recovery in bulk RNA-seq. The original code has
been optimized for sparse matrices and smoother integration with
[AnnData](https://anndata.readthedocs.io/en/latest/index.html). The NMF solver
can now be changed to the much quicker Coordinate Descent algorithm.

# Installation

This package is not yet available on [PyPi](https://pypi.org/). To install
`ecotyper` for Python, clone the repository

```bash
$ git clone https://github.com/hoseok-lee/ecotyper-py.git
```

and install with **pip**.

```bash
$ pip install .
```

# Usage

To run EcoTyper pipelines, instantiate an `EcoTyper` object and run either
`discovery()` or `recovery()`. Recovery is not yet implemented.

```python
eco = EcoTyper(adata)
eco.discovery(
    cell_type_idx,
    sample_idx,

    n_jobs              = 10,
    fractions           = "cell_type_specific",
    n_cells_gene_filter = 500,
    n_cells_distance    = 2500,
    n_genes_top_nmf     = 1000,
    cophenetic_cutoff   = 0.95,
    max_clusters        = 20,
    nmf_restarts        = 5,
    random_state        = 0,

    figures             = "output/folder"
)
```