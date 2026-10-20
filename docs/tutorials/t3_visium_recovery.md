# Tutorial 3: Recovery of cell states and ecotypes in Visium spatial data

**Goal:** map the Carcinoma cell states and ecotypes onto the spots of a 10x Visium array.
R equivalent: `EcoTyper_recovery_visium.R`.

:::{tip}
**In short, with the scverse API** (an AnnData from `sc.read_visium`, plus CIBERSORTx fractions):

```python
import ecotyper as et
adata.obsm["csx"] = fractions.loc[adata.obs_names]   # spots x cell types
et.recovery(adata, "Carcinoma", fractions="csx", n_jobs=8)
adata.obsm["ecotyper_ecotype_abundance"]; adata.obsm["ecotyper_state_abundance"]
```
:::

Each Visium spot mixes several cells, so recovery runs on every spot as a small bulk sample.
Each spot's state abundances are then weighted by the cell type fractions of the spot. Those
fractions come from CIBERSORTx, which is not part of the port: you provide them precomputed,
as the R tutorial does with `visium_fractions_example.txt`.

## Inputs

| Input | Example | Contents |
|---|---|---|
| Visium folder | `ecotyper/example_data/VisiumBreast/` | Space Ranger output: `matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`, `tissue_positions_list.csv` |
| Cell type fractions | `ecotyper/example_data/visium_fractions_example.txt` | CIBERSORTx output: `Mixture` (spot barcode) + one column per cell type (+ `P.value`, `Correlation`, `RMSE`, which are ignored) |
| Background cell type | `Epithelial.cells` | the cancer's cell of origin (Carcinoma: `Epithelial.cells`; Lymphoma: `B.cells`) |

## Step 1: Load the array and the fractions

```python
from ecotyper import workflows as wf
from ecotyper.io import read_fractions
from ecotyper.model import EcoTyperModel

expr, spots = wf.read_visium("ecotyper/example_data/VisiumBreast")
expr.shape          # genes x 3813 spots
spots.head()        # ID, X (array row), Y (array column)

frac = read_fractions("ecotyper/example_data/visium_fractions_example.txt")
model = EcoTyperModel.load("ecotyper/EcoTyper", "Carcinoma")
```

## Step 2: Recover states and ecotypes per spot

```python
states, long, wide, eco = wf.recover_visium_frame(
    expr, spots, model, frac, sample_name="VisiumBreast",
    malignant_cell="Epithelial.cells", n_jobs=8)
```

| Returned | Contents |
|---|---|
| `states` | per cell type {class}`~ecotyper.recovery_steps.StateRecovery` on the spots |
| `wide` | `state_abundances.txt`: one row per spot, `ID`, `X`, `Y`, `Sample`, `Malignant`, then one column per `<cell type>_<state>` |
| `long` | the same in long format |
| `eco` | `ecotype_abundances.txt`: one row per spot, `CE1`…`CE10` |

## Step 3: Plot an ecotype on the tissue

The R script draws spatial heatmaps; the port returns the numbers, so you can plot them with
matplotlib (`pip install matplotlib`):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(eco["Y"], -eco["X"], c=eco["CE9"], s=6, cmap="Blues")
ax.set_aspect("equal"); ax.axis("off"); ax.set_title("CE9 abundance")
fig.colorbar(sc, shrink=0.6)
```

`X`/`Y` are array row and column, as in R's plots. `Malignant` is the background cell type's
fraction per spot, which R draws as the gray background.

## Step 4 (optional): R-compatible output files

```bash
ecotyper-py recover-visium -i ecotyper/example_data/VisiumBreast \
  -f ecotyper/example_data/visium_fractions_example.txt -n VisiumBreast \
  -d Carcinoma --models ecotyper/EcoTyper -o VisiumOut -t 8
```

```text
VisiumOut/VisiumOutput/VisiumBreast/state_abundances.txt, ecotype_abundances.txt
VisiumOut/EcoTyper/Carcinoma/Carcinoma_Fractions/Cell_States/recovery/VisiumBreast/...   per cell type
```

`--background` sets the background cell type (default `Epithelial.cells`). If the column is
missing from the fractions file, the background is uniform, as in R.
