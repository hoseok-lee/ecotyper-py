# R environment for generating EcoTyper reference outputs on Linux x86-64.
# Same package versions as the Mac `ecotyper-r` conda env; installed at the same
# prefix so validation/r_reference/run_r_reference.sh works unchanged.
FROM mambaorg/micromamba:2.3-debian12
USER root
# R sorts gene names with the en_US.UTF-8 collation on the Mac; keep the same locale
RUN apt-get update && apt-get install -y --no-install-recommends locales \
 && sed -i 's/^# *en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen && rm -rf /var/lib/apt/lists/*
RUN micromamba create -y -p /opt/miniconda3/envs/ecotyper-r -c conda-forge -c bioconda \
      r-base=4.4.3 python \
      bioconductor-biobase=2.66.0 bioconductor-biocgenerics=0.52.0 bioconductor-complexheatmap=2.22.0 \
      bioconductor-iranges=2.40.0 bioconductor-s4vectors=0.44.0 \
      r-argparse=2.3.1 r-circlize=0.4.18 r-cluster=2.1.8.3 r-config=0.3.2 r-cowplot=1.2.0 \
      r-data.table=1.18.6.1 r-doparallel=1.0.17 r-fields=17.3 r-foreach=1.5.2 r-ggplot2=4.0.3 \
      r-gridbase=0.4_7 r-gridextra=2.3.1 r-matrixstats=1.5.0 r-matrixtests=0.2.3.1 \
      r-pkgmaker=0.32.10 r-plyr=1.8.9 r-rcolorbrewer=1.1_3 r-registry=0.5_1 r-reshape2=1.4.5 \
      r-rngtools=1.5.2 r-stringr=1.6.0 r-viridis=0.6.5 r-xtable=1.8_8 r-yaml=2.3.12 \
      r-digest=0.6.39 r-codetools=0.2_20 r-colorspace=2.1_3 r-spam=2.11_4 r-maps=3.4.3 r-ncdf4=1.24 \
      c-compiler cxx-compiler fortran-compiler make \
 && micromamba clean -a -y
ENV PATH=/opt/miniconda3/envs/ecotyper-r/bin:$PATH
# NMF's DESCRIPTION imports BiocManager (separate layer keeps the big env layer cached)
RUN micromamba install -y -p /opt/miniconda3/envs/ecotyper-r -c conda-forge r-biocmanager=1.30.27 && micromamba clean -a -y
# CRAN-only packages, pinned to the Mac versions
RUN Rscript -e 'for (u in c("https://cran.r-project.org/src/contrib/Archive/NMF/NMF_0.28.tar.gz", \
                            "https://cran.r-project.org/src/contrib/Archive/HiClimR/HiClimR_2.2.1.tar.gz")) { \
                  f <- tempfile(fileext = ".tar.gz"); \
                  ok <- tryCatch({download.file(u, f, quiet = TRUE); TRUE}, error = function(e) FALSE); \
                  if (!ok) download.file(sub("Archive/[^/]+/", "", u), f, quiet = TRUE); \
                  install.packages(f, repos = NULL, type = "source") }; \
                library(NMF); library(HiClimR); cat(as.character(packageVersion("NMF")), as.character(packageVersion("HiClimR")), "\n")'
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
