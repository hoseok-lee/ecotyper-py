"""data.frame semantics needed to reproduce EcoTyper's tables row for row."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .collate import sort_key


def _is_str_like(values) -> bool:
    arr = np.asarray(values, dtype=object)
    return any(isinstance(v, str) for v in arr)


def _key_array(values, decreasing: bool = False):
    """Return a list of sortable keys reproducing R's comparison for one column.
    NA (None/NaN) sorts last (na.last = TRUE) regardless of ``decreasing``."""
    arr = np.asarray(values, dtype=object)
    out = []
    if _is_str_like(arr):
        for v in arr:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append((1, b""))
            else:
                out.append((0, sort_key(str(v))))
    else:
        for v in arr:
            fv = float(v) if v is not None else np.nan
            if np.isnan(fv):
                out.append((1, 0.0))
            else:
                out.append((0, fv))
    return out


def r_order(*keys, decreasing: bool = False) -> np.ndarray:
    """``order(k1, k2, ...)`` → 0-based stable permutation.

    Numeric keys may be negated by the caller (``order(x, -y)``)."""
    n = len(keys[0])
    cols = [_key_array(k) for k in keys]

    def keyf(i):
        return tuple(c[i] for c in cols)

    idx = list(range(n))
    if decreasing:
        # R: ties keep original order; NA last
        nonna = [i for i in idx if all(c[i][0] == 0 for c in cols)]
        na = [i for i in idx if i not in set(nonna)]
        nonna.sort(key=keyf, reverse=True)
        # reverse=True in Python's sort keeps stability for equal keys
        return np.array(nonna + na, dtype=np.int64)
    idx.sort(key=keyf)
    return np.array(idx, dtype=np.int64)


def factor_levels(values) -> list:
    """``levels(factor(values))``: sorted unique non-NA values."""
    arr = [v for v in np.asarray(values, dtype=object) if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    uniq = list(dict.fromkeys(arr))
    if _is_str_like(uniq):
        return sorted((str(u) for u in uniq), key=sort_key)
    return sorted(uniq, key=float)


def r_match(x, table) -> np.ndarray:
    """``match(x, table)`` → 0-based positions, -1 for no match."""
    pos: dict = {}
    for i, v in enumerate(table):
        if v not in pos:
            pos[v] = i
    return np.array([pos.get(v, -1) for v in x], dtype=np.int64)


def make_names(names, unique: bool = False) -> list[str]:
    """``make.names(names, unique)``."""
    out = []
    for s in names:
        s = "NA" if s is None else str(s)
        s = re.sub(r"[^A-Za-z0-9._]", ".", s)
        if s == "" or not re.match(r"^([A-Za-z]|[.](?![0-9]))", s):
            s = "X" + s
        if s in _RESERVED:
            s = s + "."
        out.append(s)
    if unique:
        out = make_unique(out)
    return out


_RESERVED = {"if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE",
             "NULL", "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_character_", "NA_complex_", "in"}


def make_unique(names, sep: str = ".") -> list[str]:
    """``make.unique(names)``."""
    seen: dict[str, int] = {}
    used = set(names)
    out = []
    for s in names:
        if s not in seen:
            seen[s] = 0
            out.append(s)
            continue
        k = seen[s]
        while True:
            k += 1
            cand = f"{s}{sep}{k}"
            if cand not in used:
                break
        seen[s] = k
        used.add(cand)
        out.append(cand)
    return out


def r_merge(x: pd.DataFrame, y: pd.DataFrame, by: str, all_x: bool = False, all_y: bool = False,
            suffixes=(".x", ".y")) -> pd.DataFrame:
    """``merge(x, y, by = by, all.x, all.y)`` for a single key column.

    Rows: matched pairs, then unmatched x, then unmatched y, finally stably
    ordered by the key (``sort = TRUE``). Columns: key, remaining x columns,
    remaining y columns (duplicates suffixed). The row index is reset
    (R gives 1..n)."""
    xk = list(x[by])
    yk = list(y[by])
    ypos: dict = {}
    for j, v in enumerate(yk):
        ypos.setdefault(v, []).append(j)
    pairs = []
    matched_y = set()
    for i, v in enumerate(xk):
        for j in ypos.get(v, []):
            pairs.append((i, j))
            matched_y.add(j)
    # R's internal merge lists pairs ordered by y then x for duplicates; keys are unique here
    xi = [p[0] for p in pairs]
    yi = [p[1] for p in pairs]
    x_rest = [c for c in x.columns if c != by]
    y_rest = [c for c in y.columns if c != by]
    common = set(x_rest) & set(y_rest)
    xcols = {c: (c + suffixes[0] if c in common else c) for c in x_rest}
    ycols = {c: (c + suffixes[1] if c in common else c) for c in y_rest}
    parts = []
    m = pd.DataFrame({by: [xk[i] for i in xi]})
    for c in x_rest:
        m[xcols[c]] = x[c].iloc[xi].to_numpy() if xi else pd.Series([], dtype=x[c].dtype)
    for c in y_rest:
        m[ycols[c]] = y[c].iloc[yi].to_numpy() if yi else pd.Series([], dtype=y[c].dtype)
    parts.append(m)
    if all_x:
        ux = [i for i in range(len(xk)) if i not in set(xi)]
        if ux:
            u = pd.DataFrame({by: [xk[i] for i in ux]})
            for c in x_rest:
                u[xcols[c]] = x[c].iloc[ux].to_numpy()
            for c in y_rest:
                u[ycols[c]] = np.nan
            parts.append(u)
    if all_y:
        uy = [j for j in range(len(yk)) if j not in matched_y]
        if uy:
            u = pd.DataFrame({by: [yk[j] for j in uy]})
            for c in x_rest:
                u[xcols[c]] = np.nan
            for c in y_rest:
                u[ycols[c]] = y[c].iloc[uy].to_numpy()
            parts.append(u)
    res = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    res = res[[by] + [xcols[c] for c in x_rest] + [ycols[c] for c in y_rest]]
    o = r_order(res[by].tolist())
    return res.iloc[o].reset_index(drop=True)
