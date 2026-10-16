"""Readers for EcoTyper's user-facing inputs and conversion to AnnData.

The readers reproduce how the R scripts read the same files (``read.delim``
for bulk/Visium matrices and annotations, ``data.table::fread`` for scRNA-seq
matrices), including R's number parsers, so that a Python run starts from
bit-identical values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rcompat.frame import make_names
from .rcompat.io import read_delim
from .rcompat.parse import read_numeric_table


def read_bulk_matrix(path) -> pd.DataFrame:
    """``read.delim(path); rownames(data) = data[,1]; data = data[,-1]``.

    Returns a genes x samples DataFrame; sample names go through ``make.names``
    as in R (``check.names = TRUE``)."""
    header, rows, X = read_numeric_table(path, mode="r")
    names = make_names(header, unique=True)
    cols = names[1:] if len(names) == X.shape[1] + 1 else names[-X.shape[1]:]
    return pd.DataFrame(X, index=pd.Index(rows, dtype=object), columns=cols)


def read_scrna_matrix(path) -> pd.DataFrame:
    """``fread(path, data.table = F)``, then drop NA and duplicated gene names
    (``read_scRNA`` in lib/scRNA.R). Column names are kept verbatim."""
    header, rows, X = read_numeric_table(path, mode="fread")
    cols = header[1:] if len(header) == X.shape[1] + 1 else header[-X.shape[1]:]
    keep = []
    seen = set()
    for i, g in enumerate(rows):
        if g in ("NA", "") or g in seen:
            continue
        seen.add(g)
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    return pd.DataFrame(X[keep], index=pd.Index([rows[i] for i in keep], dtype=object), columns=cols)


def read_annotation(path) -> pd.DataFrame | None:
    """``read.delim(annotation)``; ``None`` if ``path`` is None/"NULL"/missing."""
    if path is None or str(path) == "NULL" or not Path(path).exists():
        return None
    return read_delim(path)


def read_fractions(path) -> pd.DataFrame:
    """CIBERSORTx fractions (``read_fractions`` in lib/misc.R): columns up to
    (excluding) ``P.value`` are kept."""
    df = read_delim(path)
    names = make_names(list(df.columns))
    if "P.value" in names:
        df = df.iloc[:, : names.index("P.value")]
    return df


def to_anndata(expr: pd.DataFrame, annotation: pd.DataFrame | None = None):
    """Genes x samples DataFrame (+ annotation with an ``ID`` column) → AnnData
    with samples/cells as ``obs`` and genes as ``var``."""
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index([str(c) for c in expr.columns], dtype=object))
    if annotation is not None:
        ann = annotation.copy()
        ann.index = pd.Index([str(i) for i in ann["ID"]], dtype=object)
        ann = ann[~ann.index.duplicated()]
        obs = obs.join(ann.drop(columns=["ID"]), how="left")
        obs.insert(0, "ID", obs.index.to_numpy())
    var = pd.DataFrame(index=pd.Index([str(g) for g in expr.index], dtype=object))
    return ad.AnnData(X=np.ascontiguousarray(expr.to_numpy(dtype=np.float64).T), obs=obs, var=var)


def from_anndata(adata, layer: str | None = None) -> pd.DataFrame:
    """AnnData → genes x samples DataFrame (dense)."""
    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(np.asarray(X, dtype=np.float64).T, index=pd.Index(list(adata.var_names), dtype=object),
                        columns=list(adata.obs_names))


def annotation_from_anndata(adata) -> pd.DataFrame:
    obs = adata.obs.copy()
    if "ID" not in obs.columns:
        obs.insert(0, "ID", list(adata.obs_names))
    cols = ["ID"] + [c for c in obs.columns if c != "ID"]
    out = obs[cols].reset_index(drop=True)
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
    return out
