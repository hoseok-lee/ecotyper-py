"""Text → double conversion exactly as R does it.

R does not use the C library's (correctly rounded) ``strtod``:

* ``read.delim``/``read.table``/``scan`` → ``type.convert`` → ``R_strtod5``
  (``src/main/util.c``): digits accumulated with ``ans = 10*ans + d`` (an FMA
  on arm64) and then divided by a power of ten built by repeated squaring, so
  values with more than ~22 decimal places of scale are not always correctly
  rounded.
* ``data.table::fread`` → ``parse_double_regular`` (``fread.c``): 64-bit
  integer mantissa, then one multiplication/division by a tabulated ``10^k``.

EcoTyper's scripts exchange data through text files, so reproducing R's
numbers bit for bit requires the same parsers. :func:`read_numeric_table`
parses a tab-delimited numeric table in either mode.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .fma import fma

NA_REAL = np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0]
_POW10 = np.array([float(f"1e{k}") for k in range(309)])  # correctly rounded constants


@nb.njit(cache=True)
def _r_strtod(buf, start, end, na_real):
    """Port of R_strtod5(str, NULL, '.', NA = TRUE, exact = FALSE) on buf[start:end]."""
    p = start
    while p < end and (buf[p] == 32 or buf[p] == 9):
        p += 1
    if p + 1 < end + 1 and end - p >= 2 and buf[p] == 78 and buf[p + 1] == 65:  # "NA"
        if end - p == 2:
            return na_real
    sign = 1.0
    if p < end and buf[p] == 45:  # '-'
        sign = -1.0
        p += 1
    elif p < end and buf[p] == 43:
        p += 1
    # NaN / Inf (case-insensitive)
    if end - p >= 3:
        c0 = buf[p] | 32
        c1 = buf[p + 1] | 32
        c2 = buf[p + 2] | 32
        if c0 == 110 and c1 == 97 and c2 == 110:
            return sign * np.nan
        if c0 == 105 and c1 == 110 and c2 == 102:
            return sign * np.inf
    ans = 0.0
    expn = 0
    ndigits = 0
    while p < end and 48 <= buf[p] <= 57:
        ans = fma(ans, 10.0, float(buf[p] - 48))
        p += 1
        ndigits += 1
    if p < end and buf[p] == 46:
        p += 1
        while p < end and 48 <= buf[p] <= 57:
            ans = fma(ans, 10.0, float(buf[p] - 48))
            p += 1
            ndigits += 1
            expn -= 1
    if ndigits == 0:
        return na_real
    if p < end and (buf[p] == 101 or buf[p] == 69):
        p += 1
        expsign = 1
        if p < end and buf[p] == 45:
            expsign = -1
            p += 1
        elif p < end and buf[p] == 43:
            p += 1
        n = 0
        nd = 0
        while p < end and 48 <= buf[p] <= 57:
            if n < 9999:
                n = n * 10 + (buf[p] - 48)
            p += 1
            nd += 1
        if nd == 0:
            return na_real
        expn += expsign * n
    if expn + ndigits < -300:
        for _ in range(ndigits):
            ans /= 10.0
        expn += ndigits
    p10 = 10.0
    fac = 1.0
    if expn < -307:
        n = -expn
        while n:
            if n & 1:
                fac /= p10
            n >>= 1
            p10 *= p10
        ans *= fac
    elif expn < 0:
        n = -expn
        while n:
            if n & 1:
                fac *= p10
            n >>= 1
            p10 *= p10
        ans /= fac
    elif ans != 0.0:
        n = expn
        while n:
            if n & 1:
                fac *= p10
            n >>= 1
            p10 *= p10
        ans *= fac
    return sign * ans


@nb.njit(cache=True)
def _fread_double(buf, start, end, na_real, pow10):
    """Port of data.table's parse_double_regular_core (dec = '.')."""
    p = start
    if end - p == 2 and buf[p] == 78 and buf[p + 1] == 65:
        return na_real
    if p == end:
        return na_real
    neg = False
    if buf[p] == 45:
        neg = True
        p += 1
    elif buf[p] == 43:
        p += 1
    if end - p >= 3:
        c0 = buf[p] | 32
        c1 = buf[p + 1] | 32
        c2 = buf[p + 2] | 32
        if c0 == 110 and c1 == 97 and c2 == 110:
            return np.nan
        if c0 == 105 and c1 == 110 and c2 == 102:
            return -np.inf if neg else np.inf
    acc = np.uint64(0)
    e = 0
    while p < end and buf[p] == 48:
        p += 1
    sflimit = 18
    while p < end and 48 <= buf[p] <= 57 and sflimit:
        acc = np.uint64(10) * acc + np.uint64(buf[p] - 48)
        sflimit -= 1
        p += 1
    if sflimit == 0:
        while p < end and 48 <= buf[p] <= 57:
            p += 1
            e += 1
    if p < end and buf[p] == 46:
        p += 1
        if p < end and buf[p] == 48 and acc == 0:
            k = 0
            while p + k < end and buf[p + k] == 48:
                k += 1
            p += k
            e = -k
        k = 0
        while p + k < end and 48 <= buf[p + k] <= 57 and sflimit:
            acc = np.uint64(10) * acc + np.uint64(buf[p + k] - 48)
            k += 1
            sflimit -= 1
        p += k
        e -= k
        if sflimit == 0:
            while p < end and 48 <= buf[p] <= 57:
                p += 1
    if p < end and (buf[p] == 69 or buf[p] == 101):
        p += 1
        eneg = False
        if p < end and buf[p] == 45:
            eneg = True
            p += 1
        elif p < end and buf[p] == 43:
            p += 1
        E = 0
        nd = 0
        while p < end and 48 <= buf[p] <= 57 and nd < 3:
            E = E * 10 + (buf[p] - 48)
            p += 1
            nd += 1
        e += -E if eneg else E
    r = float(acc)
    if e < -300 or e > 300:
        extra = e + 300 if e < 0 else e - 300
        r = r / pow10[-extra] if extra < 0 else r * pow10[extra]
        e -= extra
    r = r / pow10[-e] if e < 0 else r * pow10[e]
    return -r if neg else r


@nb.njit(cache=True)
def _parse_table(buf, line_starts, line_ends, ncol, skip_first, mode, na_real, pow10, out):
    nrow = line_starts.shape[0]
    for i in range(nrow):
        p = line_starts[i]
        end = line_ends[i]
        # skip first field (row name / gene), honouring quotes
        if skip_first:
            if p < end and buf[p] == 34:
                p += 1
                while p < end and buf[p] != 34:
                    p += 1
                p += 1
            while p < end and buf[p] != 9:
                p += 1
            p += 1
        for j in range(ncol):
            q = p
            while q < end and buf[q] != 9:
                q += 1
            s = p
            e = q
            if e - s >= 2 and buf[s] == 34 and buf[e - 1] == 34:
                s += 1
                e -= 1
            if mode == 0:
                out[i, j] = _r_strtod(buf, s, e, na_real) if e > s else na_real
            else:
                out[i, j] = _fread_double(buf, s, e, na_real, pow10)
            p = q + 1


def _split_lines(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    nl = np.flatnonzero(arr == 10)
    starts = np.concatenate([[0], nl + 1])
    ends = np.concatenate([nl, [len(arr)]])
    # strip \r
    cr = (ends > starts) & (arr[np.maximum(ends - 1, 0)] == 13)
    ends = ends - cr.astype(np.int64)
    keep = ends > starts
    return arr, starts[keep], ends[keep]


def _unquote(tok: bytes) -> str:
    s = tok.decode("utf-8")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].replace('""', '"').replace('\\"', '"')
    return s


def read_numeric_table(path, mode: str = "r"):
    """Parse a tab-delimited table whose first column holds names.

    Returns ``(header, first_col, values)``; ``header`` holds every header
    field (including the first-column name when present). ``mode`` is ``"r"``
    (``read.delim``) or ``"fread"``.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    arr, starts, ends = _split_lines(data)
    header = [_unquote(t) for t in data[starts[0]:ends[0]].split(b"\t")]
    body_s = starts[1:]
    body_e = ends[1:]
    first = []
    nfields = None
    for s, e in zip(body_s, body_e):
        line = data[s:e]
        tab = line.find(b"\t")
        first.append(_unquote(line[:tab] if tab >= 0 else line))
        if nfields is None:
            nfields = line.count(b"\t") + 1
    ncol = (nfields or 1) - 1
    out = np.empty((len(body_s), ncol), dtype=np.float64)
    _parse_table(arr, body_s.astype(np.int64), body_e.astype(np.int64), ncol, True,
                 0 if mode == "r" else 1, NA_REAL, _POW10, out)
    return header, first, out


def r_parse(values, mode: str = "r") -> np.ndarray:
    """Parse an iterable of strings with R's (or fread's) number parser."""
    toks = [v.encode() if isinstance(v, str) else v for v in values]
    data = b"\n".join(toks)
    arr = np.frombuffer(data, dtype=np.uint8)
    lens = np.array([len(t) for t in toks], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens + 1)[:-1]]).astype(np.int64)
    ends = starts + lens
    out = np.empty((len(toks), 1))
    _parse_table(arr, starts, ends, 1, False, 0 if mode == "r" else 1, NA_REAL, _POW10, out)
    return out[:, 0]


def text_roundtrip(x: np.ndarray, mode: str = "r") -> np.ndarray:
    """Values as R sees them after ``write.table`` (15 significant digits) and
    re-reading with ``read.delim`` (``mode="r"``) or ``fread``."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    finite = np.isfinite(flat)
    out = flat.copy()
    if finite.any():
        from .io import format_rows

        toks = format_rows(flat[finite])
        out[finite] = r_parse(toks, mode)
    return out.reshape(x.shape)
