This is an *unofficial* Python implementation for the R package
[EcoTyper](https://github.com/digitalcytometry/ecotyper/), for discovery of
ecotypes in scRNA-seq data, and recovery in bulk RNA-seq. The original code has
been optimized for sparse matrices and smoother integration with
[AnnData](https://anndata.readthedocs.io/en/latest/index.html). The NMF solver
can now be changed to the much quicker Coordinate Descent algorithm.