# EcoTyper R → Python porting plan

Source: <https://github.com/digitalcytometry/ecotyper> at commit `11408e7`
(clone in `work/ecotyper-r-src`, gitignored). Python package: `ecotyper`
(src layout). The R code is a set of command-line scripts that communicate
through tab-delimited files in a fixed directory tree; the port keeps every
computation as a pure function (pandas/numpy in, pandas/numpy out), wraps them
in AnnData-level workflow functions, and can also write the exact R directory
layout so outputs can be diffed file by file against R.

## 1. Module map

### R runtime semantics → `ecotyper.rcompat`

| R behaviour used by the pipeline | Python |
|---|---|
| `set.seed`, Mersenne-Twister `unif_rand`, `runif` | `rcompat.rng.RRandom` (R's LCG seed scrambling, MT19937 with R's state layout, `fixup`) |
| `sample()` / `sample.int()` (R ≥ 3.6 "Rejection" sampling, `R_unif_index`/`rbits`) | `RRandom.sample_int`, `RRandom.sample` |
| `hclust` (Fortran `HCLUST` + `HCASS2`, all LW methods), `cutree`, `cophenetic` | `rcompat.cluster` (line-by-line port, same loop order and tie handling) |
| `dist(method="euclidean")` | `rcompat.cluster.dist_euclidean` |
| `cluster::silhouette.default` + `summary()$avg.width` | `rcompat.cluster.silhouette` |
| `cor` (pearson, two-pass means as in `cov.c`), `var`, `sd`, `mean` (two-pass), `colMeans`, `sum` | `rcompat.stats` (sequential summation; FMA where clang contracts) |
| `scale()`, `apply(x,1,scale)` | `rcompat.stats.scale_rows` |
| `quantile(type=7)`, `median`, `mad`, `rank(ties="average")` | `rcompat.stats` |
| `p.adjust(method="BH")` | `rcompat.stats.p_adjust_bh` |
| `t.test` (Welch, two-sided) incl. "data are essentially constant" error | `rcompat.stats.welch_t_test_p` |
| `phyper(lower=TRUE)` | `rcompat.stats.phyper` (scipy hypergeom; checked against R) |
| `matrixTests::row_wilcoxon_twosample` (exact / normal approx w/ continuity + tie correction) | `rcompat.stats.row_wilcoxon_twosample` |
| `HiClimR::fastCor` | `rcompat.stats.fast_cor` |
| string collation used by `sort`, `order`, `factor`, `split`, `merge`, `dcast` (ICU in non-C locales, byte order in C) | `rcompat.collate` (PyICU / bytes), `rcompat.frame.order`, `merge`, `factor_levels` |
| `make.names`, `read.delim` (`check.names`), `write.table` | `rcompat.io` |
| `sprintf("IS%02d", factor)` uses factor *codes* | reproduced in `discovery.initial_states` |

### NMF package (v0.28) → `ecotyper.nmf`

| R | Python |
|---|---|
| `NMF::posneg` | `nmf.posneg` |
| `rnmf(rank, target)` (W then H from `runif(0, max(target))`, then `pmin`) | `nmf.rnmf` |
| `nmf(x, r, "brunet", seed=<int>, .opt="P1")` (set.seed → random seed → `divergence_update_H/W` C++ kernels, eps clamp every 10 it., `nmf.stop.connectivity(stopconv=40, check.interval=10)`, `maxIter=2000`, KL deviance) | `nmf.nmf_brunet` (numba kernels mirroring loop order/FMA) |
| `NMFfitX(list, .merge=TRUE)` (consensus = mean connectivity, best fit = first minimal deviance) | `nmf.merge_runs` |
| `cophcor`, `dispersion`, `silhouette(what=samples/features/chc/consensus)`, `predict` | `nmf.cophcor`, `nmf.dispersion`, `nmf.silhouettes`, `nmf.predict_columns` |
| `NMFpredict` (lib/ecotyper.R: fixed W, H-only Brunet update, random H from `rnmf`) | `nmf.nmf_predict` |

### `pipeline/lib/*.R` → `ecotyper.core`

| R function | Python |
|---|---|
| `misc.R::get_variable_genes` | `core.get_variable_genes` |
| `misc.R::scale_data` | `core.scale_data` |
| `misc.R::read_clinical` | `core.read_clinical` / `io.read_annotation` |
| `misc.R::collapse_samples` | `core.collapse_samples` |
| `misc.R::get_FC_for_clusters` | `core.get_fc_for_clusters` |
| `misc.R::doDE` / `doCV` | `core.do_de` / `core.do_cv` |
| `misc.R::ecotype_to_factor` | `core.ecotype_levels` |
| `misc.R::read_fractions` | `io.read_fractions` |
| `misc.R::get_colors`, `lookup_celltype*`, `theme_publication`; `heatmaps.R` | plotting only — `ecotyper.plotting` (optional, not parity-tested) |
| `ecotyper.R::NMFpredict` | `nmf.nmf_predict` |
| `ecotyper.R::averge_FC_marker_genes`, `calculate_recovery_significance` | `recovery.average_fc_marker_genes`, `recovery.recovery_significance` |
| `scRNA.R::read_scRNA` | `recovery.select_cell_type` |
| `config.R::check_discovery_configuration*` | `workflows` input validation |
| `multithreading.R` (job queue) | `concurrent.futures` process pool in `workflows` |

### `pipeline/*.R` scripts

| Script | Python |
|---|---|
| `state_recovery_bulk.R`, `state_recovery_visium.R` | `recovery.recover_states_bulk` |
| `state_recovery_scRNA.R` (+ `sum_squares` smoothing for plots) | `recovery.recover_states_scrna` |
| `ecotypes_recovery.R` | `recovery.recover_ecotypes_bulk` |
| `ecotypes_recovery_scRNA.R` | `recovery.recover_ecotypes_scrna` |
| `spatial_load_visium_data.R` | `io.read_visium` |
| `spatial_states.R`, `spatial_ecotypes.R` | `recovery.spatial_states`, `recovery.spatial_ecotypes` |
| `spatial_plot_*.R` | plotting (optional) |
| `state_discovery_extract_features.R` | `discovery.extract_features` |
| `state_discovery_NMF.R` | `discovery.run_nmf_grid` (per rank/restart) |
| `state_discovery_combine_NMF_restarts.R` | `discovery.combine_restarts` |
| `state_discovery_rank_selection.R` | `discovery.select_ranks` |
| `state_discovery_initial_plots.R` | `discovery.initial_states` |
| `state_discovery_first_filter.R` | `discovery.first_filter` |
| `state_discovery_calculate_dropout_score.R` | `discovery.dropout_scores` |
| `state_discovery_second_filter.R` | `discovery.second_filter` |
| `ecotypes.R` | `discovery.discover_ecotypes` |
| `ecotypes_assign_samples.R` | `discovery.assign_samples` |
| `state_discovery_scRNA_filter_genes.R` | `discovery_scrna.filter_genes` |
| `state_discovery_scRNA_distances.R` | `discovery_scrna.cross_correlation` |
| `state_discovery_initial_plots_scRNA.R` | `discovery.initial_states(cross_cor=True)` |
| `state_discovery_extract_features_scRNA.R` | `discovery_scrna.extract_features` |
| `state_discovery_first_filter_scRNA.R` | `discovery.first_filter` (scRNA variant) |
| `ecotypes_scRNA.R`, `ecotypes_assign_samples_scRNA.R` | `discovery_scrna.discover_ecotypes`, `discovery_scrna.assign_samples` |
| `state_discovery_presorted_filter_genes.R` | `discovery.presorted_filter_genes` |
| `csx_*.R` | **not ported** (CIBERSORTx is closed source); CSx outputs are inputs |

### Wrappers → `ecotyper.workflows`

| R wrapper | Python |
|---|---|
| `EcoTyper_recovery_bulk.R` | `workflows.recover_bulk(adata, model, ...)` |
| `EcoTyper_recovery_scRNA.R` | `workflows.recover_scrna(adata, model, ...)` |
| `EcoTyper_recovery_visium.R` | `workflows.recover_visium(adata, model, fractions, ...)` |
| `EcoTyper_discovery_bulk.R` (steps 3–8) | `workflows.discover_bulk(hires_profiles, ...)` |
| `EcoTyper_discovery_scRNA.R` | `workflows.discover_scrna(adata, ...)` |
| `EcoTyper_discovery_presorted.R` | `workflows.discover_presorted(profiles, ...)` |

`ecotyper.model.EcoTyperModel` loads/saves a discovery result in the R
`EcoTyper/<name>/<fractions>/...` layout (the prebuilt Carcinoma/Lymphoma models
ship in that layout).

## 2. Sources of nondeterminism / platform dependence

1. **`set.seed(1234 + restart)` + NMF `seed=<int>`** (`state_discovery_NMF.R`): NMF
   calls `set.seed(seed)` then the `random` seeding method, i.e. `rnmf` → W
   (`n×r`) then H (`r×p`) from `runif(max = max(target))`, clamped by `pmin`.
   Reproduced with `RRandom`.
2. **Unseeded `rnmf()` in `NMFpredict`** (all recovery scripts): R draws the
   initial H from an RNG seeded from time/PID, so R's own recovery output is not
   reproducible (rerunning R changes abundances by up to ~2e-3, discrete
   assignments unchanged). The port takes an explicit `seed`; the R reference
   is generated with `set.seed(42)` at start-up (`validation/r_reference/Rprofile.R`).
   `seed=None` reproduces R's behaviour (fresh entropy).
3. **`set.seed(1234); sample(...)`** subsampling: scRNA discovery (500 cells per
   type for DE, `split` order = sorted cell types), scRNA distances (>2500 cells),
   scRNA recovery (`-s`), presorted gene filter.
4. **`set.seed(100 + x)` + `apply(data, 1, sample)`** in the recovery
   significance bootstrap (`-z TRUE`), then unseeded continuation into `rnmf`.
5. **`set.seed(0)`** in `get_variable_genes` (no draws follow; kept for RNG-state parity).
6. **`sample(1:2, 1)`** in `heatmap_color_annotation` (plots only, after all numeric outputs).
7. **Floating point**: NMF C++ kernels are compiled with FMA contraction on arm64; R's
   `sum`/`mean` use `long double` (80-bit on x86-64 Linux, 64-bit on Apple
   silicon). Reference platform here: macOS arm64, so `long double == double`.
8. **Collation**: `sort`/`order`/`factor`/`merge`/`split`/`dcast` order character
   keys with ICU (non-C locale) or bytes (C locale). Reference runs use
   `LANG=en_US.UTF-8`; `ecotyper` defaults to the same ICU collation (`collation="C"` available).
9. **Tie-breaking**: `which.max` (first), `hclust` nearest-neighbour search
   (strict `<`/`>` in Fortran loop order), best NMF restart (first strict minimum),
   connectivity stopping (compares connectivity matrices, i.e. partitions).
10. **Parallelism**: R runs scripts in parallel processes; results are
    order-independent because every process reseeds or is unseeded.

## 3. Validation strategy

* Unit parity vs R (small inputs, generated by `tests/r_scripts/*.R`): RNG streams,
  `sample`, `rnmf`, single Brunet iterations and full runs (W, H, iterations,
  deviance), NMFpredict, hclust/cutree/cophenetic, cophcor, silhouettes,
  t-test/BH/phyper/wilcoxon, scaling, variable genes.
* End-to-end per tutorial: run the original R wrapper (commands in
  `validation/r_reference/run_r_reference.sh`), run the Python workflow, diff
  every text output and the intermediate `EcoTyper/...` files.
