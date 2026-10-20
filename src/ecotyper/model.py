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
    """A discovery result: chosen ranks, per-cell-type states and ecotypes.

    Get one from :func:`ecotyper.discovery`, :meth:`load` (a saved model, an
    R discovery folder, or a built-in model by name: ``"Carcinoma"``,
    ``"Lymphoma"``) and pass it to :func:`ecotyper.recovery`. :meth:`save`
    writes the files recovery needs in the R layout, so a saved model can be
    used by both this package and the R scripts.
    """

    name: str
    fractions: str
    rank_data: pd.DataFrame  # CellType, Chosen_Rank (file order)
    cell_types: dict = field(default_factory=dict)  # cell type -> CellTypeModel
    ecotypes: pd.DataFrame | None = None  # ID, CellType, State, Ecotype (file order)
    path: Path | None = None
    #: full in-memory discovery result (:class:`ecotyper.workflows.DiscoveryResult`) when the
    #: model comes from :func:`ecotyper.discovery`; not saved
    discovery_result: object | None = field(default=None, repr=False)

    @property
    def cell_type_order(self) -> list[str]:
        return [str(c) for c in self.rank_data.iloc[:, 0]]

    @property
    def n_states(self) -> dict:
        """Cell type → number of cell states."""
        return {ct: m.n_states for ct, m in self.cell_types.items()}

    @property
    def ecotype_names(self) -> list[str]:
        if self.ecotypes is None:
            return []
        return ecotype_levels(self.ecotypes["Ecotype"])

    def __repr__(self) -> str:
        states = ", ".join(f"{ct} ({n})" for ct, n in self.n_states.items())
        src = f", path={str(self.path)!r}" if self.path is not None else ""
        return (f"EcoTyperModel(name={self.name!r}, {len(self.cell_types)} cell types, "
                f"{sum(self.n_states.values())} states, {len(self.ecotype_names)} ecotypes{src})\n"
                f"    states: {states}")

    @classmethod
    def from_discovery(cls, result, name: str = "custom", fractions: str | None = None) -> "EcoTyperModel":
        """Model from an in-memory discovery result (``discover_from_profiles`` /
        ``discover_scrna_frame``).

        The R pipeline hands discovery results to recovery through text files
        (15 significant digits). To stay exact, the model tables take the same
        text round trip here (through a temporary folder that is removed
        again), so recovery with this model equals recovery with the model
        written by the file-based workflow."""
        import tempfile

        if fractions is None:
            fractions = _default_fractions(result)
        with tempfile.TemporaryDirectory(prefix="ecotyper_model_") as tmp:
            base = Path(tmp) / name / fractions
            write_model_files(result, base, heatmap=False)
            model = cls.load(base)
        model.path = None
        for ctm in model.cell_types.values():
            ctm.path = None
        model.discovery_result = result
        return model

    def save(self, path, *, heatmap: bool = True) -> Path:
        """Write the model to ``path`` (the R ``<discovery>/<fractions>`` folder layout).

        Load it back with ``EcoTyperModel.load(path)``. With ``heatmap`` the
        (large) ``heatmap_data.txt``/``heatmap_top_ann.txt`` files that the R
        recovery scripts use for their heatmap plots are written too, when
        available."""
        import shutil

        path = Path(path)
        if self.discovery_result is not None:
            write_model_files(self.discovery_result, path, heatmap=heatmap)
            return path
        if self.path is None:
            raise ValueError("This model has neither a source folder nor a discovery result to save from")
        src = Path(self.path)
        if src.resolve() == path.resolve():
            return path
        files = ["Analysis/rank_selection/rank_data.txt", "Ecotypes/discovery/ecotypes.txt",
                 "Ecotypes/discovery/mapping_all_states.txt"]
        state_files = ["W.txt", "mapping_to_initial_states.txt", "initial_gene_info.txt", "gene_info.txt"]
        if heatmap:
            state_files += ["heatmap_data.txt", "heatmap_top_ann.txt"]
        for ct, m in self.cell_types.items():
            files += [f"Cell_States/discovery/{ct}/{m.n_states}/{f}" for f in state_files]
        for f in files:
            if (src / f).exists():
                (path / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / f, path / f)
        return path

    @classmethod
    def load(cls, root, discovery: str | None = None, fractions: str | None = None) -> "EcoTyperModel":
        """Load a model.

        * ``load("Carcinoma")`` / ``load("Lymphoma")``: a built-in model (see
          :mod:`ecotyper.builtin`; downloaded and cached on first use);
        * ``load(path)``: a folder written by :meth:`save` or an R
          ``EcoTyper/<discovery>/<fractions>`` folder (contains ``Analysis/``);
        * ``load(root, discovery, fractions)``: ``<root>/<discovery>/<fractions>``
          (``fractions`` defaults to ``<discovery>_Fractions``)."""
        from . import builtin

        if discovery is None and str(root) in builtin.BUILTIN and not Path(root).exists():
            root = builtin.builtin_model_path(str(root))
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
    elif gi is not None:
        # heatmap_data.txt rows are the genes of gene_info.txt (first_filter writes both
        # from the same table), so models saved without the large heatmap file still work
        hm = [str(g) for g in gi["Gene"]]
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


def _default_fractions(result) -> str:
    """Folder name the file-based workflows use for a discovery result."""
    if result.get("cross_cor") is None:
        return "All_genes"
    fg = result["params"].get("filter_genes", "cell type specific")
    return {"cell type specific": "Cell_type_specific_genes", "no filter": "All_genes"}.get(fg, f"Top_{fg}")


def model_ecotypes_table(result) -> pd.DataFrame:
    """``ecotypes.txt`` as written by the discovery workflow (the scRNA variant
    has R's extra ``Cell type`` column, ecotypes_scRNA.R:152)."""
    et = result["ecotypes"].ecotypes
    if result.get("cross_cor") is not None and "Cell type" not in et.columns:
        et = et.copy()
        et.insert(list(et.columns).index("Ecotype"), "Cell type", et["CellType"])
    return et


def write_model_files(result, base, *, heatmap: bool = True) -> Path:
    """Write the files recovery reads from a discovery result into ``base``
    (``<discovery>/<fractions>``), byte-identical to the corresponding files of
    :func:`ecotyper.workflows.write_discovery` / ``write_discovery_scrna``."""
    from .rcompat.io import write_table
    from .rlayout import _wm

    base = Path(base)
    write_table(result["ranks"], base / "Analysis" / "rank_selection" / "rank_data.txt", row_names=False)
    for ct, ini in result["initial"].items():
        d = base / "Cell_States" / "discovery" / ct / str(ini.rank)
        _wm(ini.W, d / "W.txt")
        write_table(ini.initial_gene_info, d / "initial_gene_info.txt")
        st = result["states"][ct]
        write_table(st.mapping, d / "mapping_to_initial_states.txt", row_names=False)
        write_table(st.gene_info, d / "gene_info.txt")
        if heatmap:
            write_table(st.heatmap_top_ann, d / "heatmap_top_ann.txt")
            _wm(st.heatmap_data, d / "heatmap_data.txt")
    ed = base / "Ecotypes" / "discovery"
    write_table(result["ecotypes"].mapping_all_states, ed / "mapping_all_states.txt", row_names=False)
    write_table(model_ecotypes_table(result), ed / "ecotypes.txt", row_names=False)
    return base
