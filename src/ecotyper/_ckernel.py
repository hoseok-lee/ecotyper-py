"""Build and load the C kernels shipped as source (``_nmfk.c``) with the system C compiler.

The R-exact Brunet loop (:func:`ecotyper.nmf.run_brunet`) uses ``_nmfk.c`` when it can be
compiled: same operations as the numba kernels, blocked so that the compiler vectorises them
(bit-identical results, verified in ``tests/test_nmf.py``). ``ECOTYPER_NMF_KERNEL=numba``
forces the numba kernels. Built libraries are cached in ``$ECOTYPER_CACHE`` (default: the
temporary directory).
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from .rcompat._ld import _compiler
from .rcompat.fma import USE_FMA

_SRC = Path(__file__).with_name("_nmfk.c")


def _avx512() -> bool:
    try:
        with open("/proc/cpuinfo") as fh:
            return any(line.startswith("flags") and " avx512f" in line for line in fh)
    except OSError:
        return False


def _flags() -> list[str]:
    f = ["-O3", "-ffp-contract=off", "-fno-math-errno", "-fPIC", "-shared"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        # AVX2/AVX-512 and an inlined fma(); cached per machine below. Block width = columns
        # processed together: 4 x BW accumulators should fill the vector registers (measured
        # on a Xeon Gold 6258R: BW=64 with 512-bit vectors 3.5-4x faster than the numba kernel).
        f.append("-march=native")
        f += ["-mprefer-vector-width=512", "-DBW=64"] if _avx512() else ["-DBW=32"]
    else:
        f.append("-DBW=16")  # Apple silicon: 32 x 128-bit registers
    if USE_FMA:
        f.append("-DUSE_FMA")
    return f


def build(src: Path, flags: list[str]) -> Path:
    """Compile ``src`` into a shared library (cached by source, flags and machine)."""
    cc = _compiler()
    if cc is None:
        raise RuntimeError("no C compiler found")
    key = src.read_bytes() + " ".join([sys.platform, platform.machine(), platform.node(), cc] + flags).encode()
    tag = hashlib.sha1(key).hexdigest()[:12]
    cache = Path(os.environ.get("ECOTYPER_CACHE", Path(tempfile.gettempdir()) / "ecotyper_ld"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"{src.stem}_{tag}.so"
    if so.exists():
        return so
    tmp = cache / f"{src.stem}_{tag}.{os.getpid()}.so"
    r = subprocess.run([cc, *flags, str(src), "-o", str(tmp), "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"building {src.name} failed:\n{r.stderr}")
    os.replace(tmp, so)
    return so


def _build() -> Path:
    return build(_SRC, _flags())


def _load():
    if os.environ.get("ECOTYPER_NMF_KERNEL", "c") == "numba":
        return None
    try:
        lib = C.CDLL(str(_build()))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"C NMF kernel unavailable ({e}); using the (slower) numba kernels")
        return None
    dp = C.POINTER(C.c_double)
    i64 = C.c_int64
    lib.brunet_loop.restype = i64
    lib.brunet_loop.argtypes = [dp, dp, dp, dp, i64, i64, i64, i64, C.c_int, i64, i64, C.c_double, C.POINTER(i64)]
    return lib


_LIB = None
_LOADED = False


def nmf_lib():
    """The loaded ``_nmfk`` library, or ``None`` (numba fallback)."""
    global _LIB, _LOADED
    if not _LOADED:
        _LIB = _load()
        _LOADED = True
    return _LIB
