"""Readers/writers for the tab-delimited files EcoTyper's R scripts exchange.

* :func:`read_delim` mimics ``read.delim`` (header, ``check.names``, automatic
  row names when the header has one field less, ``type.convert``).
* :func:`write_table` mimics ``write.table(x, sep = "\\t")`` including R's
  number formatting (15 significant digits, fixed vs. scientific choice), so
  files written by the port can be diffed byte for byte with R's.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .frame import make_names


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def _convert_column(values: list[str]):
    """R's ``type.convert(as.is = TRUE)`` for one column of strings."""
    na = {"NA", ""}
    nonna = [v for v in values if v not in na]
    if not nonna:
        return np.array([np.nan] * len(values))  # logical NA in R
    if all(v in ("TRUE", "FALSE", "T", "F", "true", "false", "True", "False") for v in nonna):
        return np.array([None if v in na else v in ("TRUE", "T", "true", "True") for v in values], dtype=object)
    try:
        out = np.array([np.nan if v in na else float(v) for v in values], dtype=np.float64)
        # reject things like "nan"/"inf" spelled differently than R accepts
        for v in nonna:
            lv = v.lower()
            if lv in ("nan", "inf", "-inf", "infinity", "-infinity") and v not in ("NaN", "Inf", "-Inf"):
                raise ValueError
        return out
    except ValueError:
        return np.array([None if v == "NA" else v for v in values], dtype=object)


def read_delim(path, check_names: bool = True, row_names_col: bool | None = None,
               numeric_only: bool = False) -> pd.DataFrame:
    """``read.delim(path)``.

    Row names (if the header has one field less than the data, or
    ``row_names_col=True``) become the index as strings; otherwise the index is
    ``"1".."n"`` like R's automatic row names.
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        rows = [r for r in reader]
    rows = [r for r in rows if r]  # blank.lines.skip
    header = rows[0]
    body = rows[1:]
    ncol = max((len(r) for r in body), default=len(header))
    has_rn = row_names_col if row_names_col is not None else (len(header) == ncol - 1)
    if has_rn:
        if len(header) == ncol:
            header = header[1:]
        index = [r[0] for r in body]
        body = [r[1:] for r in body]
    else:
        index = [str(i + 1) for i in range(len(body))]
    if check_names:
        header = make_names(header, unique=True)
    cols = {}
    for j, h in enumerate(header):
        vals = [r[j] if j < len(r) else "NA" for r in body]
        cols[h] = vals if False else _convert_column(vals)
    df = pd.DataFrame(cols, index=pd.Index(index, dtype=object))
    if numeric_only:
        df = df.astype(np.float64)
    return df


def read_matrix(path, check_names: bool = True) -> pd.DataFrame:
    """Read a numeric matrix written by ``write.table`` (row names + header)."""
    df = pd.read_csv(path, sep="\t", quotechar='"', index_col=0, dtype={0: str},
                     keep_default_na=True, na_values=["NA"], low_memory=False)
    df.index = df.index.astype(str)
    if check_names:
        df.columns = make_names(list(df.columns), unique=True)
    return df.astype(np.float64)


def read_expression(path, check_names: bool = False, dedup_first_col: bool = True) -> pd.DataFrame:
    """Read an expression matrix with gene names in the first column
    (``fread(path, data.table = F)``; ``check_names=True`` emulates read.delim)."""
    df = pd.read_csv(path, sep="\t", quotechar='"', low_memory=False, na_values=["NA", ""])
    first = df.columns[0]
    genes = df[first]
    df = df.drop(columns=first)
    df.index = pd.Index(genes.astype(object).where(genes.notna(), None))
    if check_names:
        df.columns = make_names(list(df.columns), unique=True)
    return df


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------
_TBL = [float(10**k) for k in range(23)]  # format.c tbl[] (double build, KP_MAX 22)
_KP_MAX = 22
_R_DEC_MIN_EXPONENT = -308


def _scientific(x: float, digits: int = 15):
    """R's ``scientific()`` (src/main/format.c) for a build where long double
    is double (macOS arm64): returns ``(neg, kpower, nsig, roundingwidens)``.

    The number of significant digits is decided by scaling and rounding in
    double precision, which differs from correctly rounded decimal conversion
    in rare half-way cases (e.g. 0.2527597964472405 is written with 14 digits).
    """
    if x == 0.0:
        return 0, 0, 1, False
    neg = 1 if x < 0 else 0
    r = -x if neg else x
    kp = int(math.floor(math.log10(r))) - digits + 1
    r_prec = r
    if abs(kp) <= _KP_MAX:
        if kp >= 0:
            r_prec /= _TBL[kp]
        else:
            r_prec *= _TBL[-kp]
    elif kp <= _R_DEC_MIN_EXPONENT:
        r_prec = (r_prec * 1e303) / math.pow(10.0, float(kp + 303))
    else:
        r_prec /= math.pow(10.0, float(kp))
    if r_prec < _TBL[digits - 1]:
        r_prec *= 10.0
        kp -= 1
    alpha = float(np.rint(r_prec))
    nsig = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1
    rgt = digits - kpower
    rgt = 0 if rgt < 0 else (_KP_MAX if rgt > _KP_MAX else rgt)
    fuzz = 0.5 / _TBL[rgt]
    widens = kpower > 0 and kpower <= _KP_MAX and r < _TBL[kpower] - fuzz
    return neg, kpower, nsig, widens


def _sci_parts(x: float, digits: int = 15):
    """Return (nsig, kpower) as R's ``scientific()`` for |x| > 0."""
    _, kp, nsig, _ = _scientific(x, digits)
    return nsig, kp


def format_real(x: float, digits: int = 15) -> str:
    """Format one double as ``write.table`` does (``formatReal`` + ``EncodeReal0``
    with ``digits = 15``)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        if isinstance(x, float) and math.isnan(x) and np.isnan(x):
            return "NA" if _is_r_na(x) else "NaN"
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    neg, kpower, nsig, widens = _scientific(x, digits)
    left = kpower + 1
    if widens:
        left -= 1
    sleft = neg + (1 if left <= 0 else left)
    rgt = max(0, nsig - left)
    mxsl = 1 + neg if left < 0 else sleft
    wF = mxsl + rgt + (rgt != 0)
    e = 2 if (left > 100 or left <= -99) else 1
    d = nsig - 1
    wE = neg + (d > 0) + d + 4 + e
    if wF <= wE:
        return f"{x:.{rgt}f}"
    s = f"{x:.{d}e}"
    mant, ex = s.split("e")
    sign = ex[0]
    ev = ex[1:].lstrip("0").rjust(e + 1, "0")
    return f"{mant}e{sign}{ev}"


_NA_BITS = 1954  # R's NA_real_ payload (lower word)


def _is_r_na(x: float) -> bool:
    b = np.array([x], dtype=np.float64).view(np.uint64)[0]
    return int(b & 0xFFFFFFFF) == _NA_BITS


def _fmt_cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "NA"
        return format_real(float(v))
    if isinstance(v, str):
        return '"' + v.replace('"', '\\"') + '"'
    if pd.isna(v):
        return "NA"
    return '"' + str(v) + '"'


def write_table(df: pd.DataFrame, path, row_names: bool = True, col_names: bool = True,
                quote_numeric_index: bool = True) -> None:
    """``write.table(df, path, sep = "\\t", row.names = row_names)``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if col_names:
        lines.append("\t".join('"' + str(c) + '"' for c in df.columns))
    cols = [df[c].to_numpy(dtype=object) for c in df.columns]
    idx = [str(i) for i in df.index]
    for r in range(len(df)):
        cells = [_fmt_cell(c[r]) for c in cols]
        if row_names:
            cells = ['"' + idx[r] + '"'] + cells
        lines.append("\t".join(cells))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


def write_matrix(mat: np.ndarray, rows, cols, path) -> None:
    """``write.table(matrix_with_dimnames, sep = "\\t")``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(mat)
    with open(path, "w") as fh:
        fh.write("\t".join('"' + str(c) + '"' for c in cols) + "\n")
        for i, rn in enumerate(rows):
            fh.write('"' + str(rn) + '"\t' + "\t".join(_fmt_cell(v) for v in mat[i]) + "\n")
