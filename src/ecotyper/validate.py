"""Compare output trees of the Python port with the R pipeline's."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .rcompat.parse import r_parse


def _read_cells(path):
    with open(path, newline="") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [ln.split("\t") for ln in lines]


def _is_num(tok: str) -> bool:
    if tok in ("NA", "NaN", "Inf", "-Inf") or tok.startswith('"') or tok == "":
        return False
    try:
        float(tok)
        return True
    except ValueError:
        return False


def compare_files(py_path, r_path) -> dict:
    """Cell-by-cell comparison of two tab-delimited files written by write.table.

    Numeric cells are compared as numbers (max abs / max rel difference);
    everything else (names, labels, NA) must be identical."""
    a = _read_cells(py_path)
    b = _read_cells(r_path)
    out = {"identical_bytes": Path(py_path).read_bytes() == Path(r_path).read_bytes(),
           "shape_equal": len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b)),
           "rows": len(b) - 1, "cols": max((len(x) for x in b), default=0)}
    if not out["shape_equal"]:
        out.update(discrete_equal=False, n_numeric=0, max_abs=np.nan, max_rel=np.nan, n_diff_numeric=np.nan,
                   n_diff_other=np.nan)
        return out
    xa, xb = [], []
    other = 0
    for ra, rb in zip(a, b):
        for ta, tb in zip(ra, rb):
            na, nb = _is_num(ta), _is_num(tb)
            if na and nb:
                xa.append(ta)
                xb.append(tb)
            elif ta != tb:
                other += 1
    if xa:
        va = r_parse(xa)
        vb = r_parse(xb)
        d = np.abs(va - vb)
        rel = d / np.maximum(np.abs(vb), np.finfo(float).tiny)
        rel[d == 0] = 0.0
        out.update(n_numeric=len(xa), max_abs=float(d.max()), max_rel=float(rel.max()),
                   n_diff_numeric=int((va != vb).sum()))
    else:
        out.update(n_numeric=0, max_abs=0.0, max_rel=0.0, n_diff_numeric=0)
    out["n_diff_other"] = other
    out["discrete_equal"] = other == 0
    return out


def compare_trees(py_root, r_root, pattern: str = ".txt") -> pd.DataFrame:
    """Compare every ``*.txt`` under ``r_root`` with the same path under ``py_root``."""
    rows = []
    r_root = Path(r_root)
    for root, _, files in sorted(os.walk(r_root)):
        for f in sorted(files):
            if not f.endswith(pattern):
                continue
            rp = Path(root) / f
            rel = rp.relative_to(r_root)
            pp = Path(py_root) / rel
            if not pp.exists():
                rows.append({"file": str(rel), "missing": True})
                continue
            res = compare_files(pp, rp)
            res["file"] = str(rel)
            res["missing"] = False
            rows.append(res)
    cols = ["file", "missing", "identical_bytes", "shape_equal", "discrete_equal", "n_numeric", "n_diff_numeric",
            "max_abs", "max_rel", "n_diff_other", "rows", "cols"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]
