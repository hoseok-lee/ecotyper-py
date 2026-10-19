# ecotyper: EcoTyper in Python

A Python/AnnData port of [EcoTyper](https://github.com/digitalcytometry/ecotyper), the
framework for identifying **cell states** and multicellular communities (**ecotypes**) in
bulk, single-cell and spatial transcriptomes (Luca, Steen et al., *Cell* 2021).

The port reproduces the R implementation. On all six README tutorials every discrete output
(states, ecotypes, ranks, gene sets) is identical to R's, and every numeric value agrees to
within 8e-11 relative. See [Benchmarks](benchmarks.md) for the comparison and timings.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Set up the environment and the R repository's example data and models.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
Recover cell states and ecotypes in an AnnData object in five lines.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Step by step through the six EcoTyper tutorials on the bundled data.
:::

:::{grid-item-card} Benchmarks
:link: benchmarks
:link-type: doc
Python vs R: wall time, memory and agreement, as charts.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
concepts
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

cli
api
benchmarks
validation
```

## Credit and license

EcoTyper was developed by Bogdan A. Luca, Chloé B. Steen, Aaron M. Newman and colleagues.
If you use it, cite:

- Luca/Steen et al., *Atlas of clinically distinct cell states and ecosystems across human
  solid tumors*, Cell 2021, <https://doi.org/10.1016/j.cell.2021.09.014>
- Steen/Luca et al., *The landscape of tumor cell states and ecosystems in diffuse large B cell
  lymphoma*, Cancer Cell 2021, <https://doi.org/10.1016/j.ccell.2021.08.011>

This port is a derivative of EcoTyper, distributed under the same **Stanford non-commercial
license**: free for non-commercial use only. It is an independent port, not affiliated
with or endorsed by the EcoTyper authors. CIBERSORTx is closed source and is not part of
the port. Steps that need it take precomputed CIBERSORTx outputs as inputs.
