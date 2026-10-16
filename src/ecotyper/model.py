"""EcoTyper discovery results ("models") in the R directory layout.

A model is the folder ``EcoTyper/<discovery>/<fractions>/`` written by the R
discovery wrappers (the prebuilt Carcinoma and Lymphoma models ship in this
layout) or by :mod:`ecotyper.workflows`. Only the files used by recovery are
loaded; numbers are parsed with R's own ``read.delim`` number parser so that
recovery starts from exactly the values R sees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .rcompat.io import read_delim
from .rcompat.parse import read_numeric_table


@dataclass
class CellTypeModel:
    """Cell states of one cell type (``Cell_States/discovery/<ct>/<rank>``)."""

    cell_type: str
    n_states: int
    W: pd.DataFrame  # (2 * genes) x initial states, rows "<gene>__pos" / "<gene>__neg"
    mapping: pd.DataFrame  # State, InitialState
    initial_gene_info: pd.DataFrame
    gene_info: pd.DataFrame | None = None
    heatmap_genes: list | None = None  # row names of heatmap_data.txt
    path: Path | None = None


@dataclass
class EcoTyperModel:
    """A discovery result: chosen ranks, per-cell-type states and ecotypes."""

    name: str
    fractions: str
    rank_data: pd.DataFrame  # CellType, Chosen_Rank (file order)
    cell_types: dict = field(default_factory=dict)  # cell type -> CellTypeModel
    ecotypes: pd.DataFrame | None = None  # ID, CellType, State, Ecotype (file order)
    path: Path | None = None

    @property
    def cell_type_order(self) -> list[str]:
        return [str(c) for c in self.rank_data.iloc[:, 0]]

    @classmethod
    def load(cls, root, discovery: str | None = None, fractions: str | None = None) -> "EcoTyperModel":
        """Load ``<root>/<discovery>/<fractions>`` (or ``root`` itself if it is
        the fractions folder, i.e. contains ``Analysis/``)."""
        root = Path(root)
        if discovery is not None:
            if fractions is None:
                fractions = f"{discovery}_Fractions"
            base = root / discovery / fractions
        else:
            base = root
            fractions = base.name
            discovery = base.parent.name
        rank = read_delim(base / "Analysis" / "rank_selection" / "rank_data.txt")
        model = cls(name=discovery, fractions=fractions, rank_data=rank, path=base)
        for ct, n in zip(rank.iloc[:, 0], rank.iloc[:, 1]):
            ct = str(ct)
            n = int(n)
            model.cell_types[ct] = load_cell_type(base / "Cell_States" / "discovery" / ct / str(n), ct, n)
        eco = base / "Ecotypes" / "discovery" / "ecotypes.txt"
        if eco.exists():
            model.ecotypes = read_delim(eco)
        return model


def read_W(path) -> pd.DataFrame:
    """``read.delim("W.txt")`` with R's number parser."""
    header, rows, vals = read_numeric_table(path, mode="r")
    cols = header[-vals.shape[1]:] if vals.shape[1] else []
    return pd.DataFrame(vals, index=pd.Index(rows, dtype=object), columns=cols)


def load_cell_type(d: Path, cell_type: str, n_states: int) -> CellTypeModel:
    d = Path(d)
    W = read_W(d / "W.txt")
    mapping = read_delim(d / "mapping_to_initial_states.txt")
    igi = read_delim(d / "initial_gene_info.txt")
    gi = read_delim(d / "gene_info.txt") if (d / "gene_info.txt").exists() else None
    hm = None
    if (d / "heatmap_data.txt").exists():
        with open(d / "heatmap_data.txt") as fh:
            next(fh)
            hm = [line.split("\t", 1)[0].strip('"') for line in fh if line.strip()]
    return CellTypeModel(cell_type=cell_type, n_states=n_states, W=W, mapping=mapping,
                         initial_gene_info=igi, gene_info=gi, heatmap_genes=hm, path=d)


def ecotype_levels(values) -> list[str]:
    """Levels of ``ecotype_to_factor(values)`` (lib/misc.R)."""
    from .rcompat.frame import factor_levels

    unq = sorted({str(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))})
    for prefix in ("CE", "E"):
        nums = []
        ok = True
        for u in unq:
            s = u.replace(prefix, "")
            try:
                nums.append(int(s))
            except ValueError:
                ok = False
                break
        if ok:
            return [f"{prefix}{k}" for k in sorted(nums)]
    return factor_levels(list(unq))
