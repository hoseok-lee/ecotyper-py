# Command line

`ecotyper-py` mirrors the R wrapper scripts and writes the same output folder layout, so it can
replace `Rscript EcoTyper_*.R` in existing pipelines. Unlike the R scripts, it takes options on
the command line rather than in a YAML config file.

| Command | R script | Tutorial |
|---|---|---|
| `recover-bulk` | `EcoTyper_recovery_bulk.R` | [1](tutorials/t1_bulk_recovery.md) |
| `recover-scrna` | `EcoTyper_recovery_scRNA.R` | [2](tutorials/t2_scrna_recovery.md) |
| `recover-visium` | `EcoTyper_recovery_visium.R` (precomputed fractions) | [3](tutorials/t3_visium_recovery.md) |
| `discover` | `EcoTyper_discovery_bulk.R` (from CIBERSORTx HiRes profiles), `EcoTyper_discovery_presorted.R` | [4](tutorials/t4_bulk_discovery.md), [6](tutorials/t6_presorted_discovery.md) |
| `discover-scrna` | `EcoTyper_discovery_scRNA.R` | [5](tutorials/t5_scrna_discovery.md) |

`-t/--threads` sets the number of parallel processes (R: `Number of threads`). Set
`OMP_NUM_THREADS=1` so that BLAS does not oversubscribe the cores.

## `ecotyper-py`

```{literalinclude} _generated/cli_main.txt
:language: text
```

## `recover-bulk`

```{literalinclude} _generated/cli_recover-bulk.txt
:language: text
```

## `recover-scrna`

```{literalinclude} _generated/cli_recover-scrna.txt
:language: text
```

## `recover-visium`

```{literalinclude} _generated/cli_recover-visium.txt
:language: text
```

## `discover`

```{literalinclude} _generated/cli_discover.txt
:language: text
```

## `discover-scrna`

```{literalinclude} _generated/cli_discover-scrna.txt
:language: text
```
