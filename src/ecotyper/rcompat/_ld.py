"""ctypes binding for ``_ld.c``: R's ``long double`` reductions on x86-64.

R accumulates ``sum``/``mean``/``colMeans``/``cov``/``cor``/``scale`` in 80-bit
x87 ``long double``. NumPy/numba only have 64-bit doubles, so on x86-64 the
sums differ from R in the last bits. ``_ld.c`` reproduces R's exact sequence of
operations; it is compiled on first use with the system C compiler.

``LD`` is the loaded library, or ``None`` where ``long double`` is not wider
than ``double`` (e.g. Apple silicon, where the numba kernels are already exact)
or when ``ECOTYPER_LD=0``. ``ECOTYPER_LD=1`` forces the C kernels.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("_ld.c")


def _wanted() -> bool:
    env = os.environ.get("ECOTYPER_LD", "auto")
    if env == "0":
        return False
    if env == "1":
        return True
    return np.finfo(np.longdouble).nmant > 52


def _compiler() -> str | None:
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    return None


def _build() -> Path:
    cc = _compiler()
    if cc is None:
        raise RuntimeError("no C compiler found (needed for R long-double kernels)")
    src = _SRC.read_bytes()
    tag = hashlib.sha1(src + sys.platform.encode()).hexdigest()[:12]
    cache = Path(os.environ.get("ECOTYPER_CACHE", Path(tempfile.gettempdir()) / "ecotyper_ld"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"_ld_{tag}.so"
    if so.exists():
        return so
    tmp = cache / f"_ld_{tag}.{os.getpid()}.so"
    base = [cc, "-O2", "-ffp-contract=off", "-fPIC", "-shared", str(_SRC), "-o", str(tmp), "-lm"]
    for extra in (["-fopenmp"], []):
        r = subprocess.run(base[:1] + extra + base[1:], capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, so)
            return so
    raise RuntimeError(f"building _ld.c failed:\n{r.stderr}")


def _load():
    if not _wanted():
        return None
    # the cor kernel is OpenMP-parallel; keep it single-threaded unless asked
    # (the workflows already run one process per task)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        lib = C.CDLL(str(_build()))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"R long-double kernels unavailable ({e}); results may differ from R in the last bits")
        return None
    dp = C.POINTER(C.c_double)
    i64 = C.c_int64
    sigs = {
        "ld_sum": (C.c_double, [dp, i64]),
        "ld_mean": (C.c_double, [dp, i64]),
        "ld_row_mean": (None, [dp, i64, i64, C.c_int, dp]),
        "ld_col_sums": (None, [dp, i64, i64, dp]),
        "ld_col_means": (None, [dp, i64, i64, dp]),
        "ld_row_means": (None, [dp, i64, i64, dp]),
        "ld_var": (C.c_double, [dp, i64]),
        "ld_row_var": (None, [dp, i64, i64, dp]),
        "ld_cor_cols": (None, [dp, i64, i64, dp]),
        "ld_cor2": (C.c_double, [dp, dp, i64]),
        "ld_scale_vec": (None, [dp, i64, dp]),
        "ld_scale_rows": (None, [dp, i64, i64, dp]),
        "ld_col_normalize": (None, [dp, i64, i64, C.c_int, dp]),
    }
    for name, (res, args) in sigs.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    return lib


LD = _load()


def ptr(a: np.ndarray):
    return a.ctypes.data_as(C.POINTER(C.c_double))
