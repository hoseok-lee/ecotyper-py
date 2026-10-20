FROM mambaorg/micromamba:2.3-debian12
RUN micromamba install -y -n base -c conda-forge \
      python=3.12.15 numpy=2.5.4 scipy=1.18.1 numba=0.68.0 llvmlite=0.50.0 \
      anndata=0.13.4 pandas=3.0.6 h5py=3.16.0 pyyaml=6.0.3 pyicu=2.16.2 pytest=9.1.1 scikit-learn=1.9.1 \
 && micromamba clean -a -y
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
USER mamba
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
