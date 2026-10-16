"""Fused multiply-add for numba kernels.

R and its packages are compiled with ``-O2`` by clang/gfortran, which contract
``a*b + c`` inside one expression into a single fused multiply-add on arm64
(verified by disassembling ``NMF.so`` and ``stats.so``: e.g. NMF's
``divergence_update_H``, ``stats::corcov`` and ``hclust_``). Reproducing R
bit-for-bit therefore requires FMAs at exactly those places; plain numba code
never contracts, so kernels call :func:`fma` explicitly where R's binaries do.

On platforms where R is not compiled with FMA contraction (e.g. x86-64 builds
without ``-mfma``) set ``ECOTYPER_NO_FMA=1`` to use ``a*b + c`` instead.
"""
from __future__ import annotations

import os

import numba as nb
from llvmlite import ir
from numba.core import types
from numba.extending import intrinsic

USE_FMA = os.environ.get("ECOTYPER_NO_FMA", "0") != "1"


@intrinsic
def _fma_intrinsic(typingctx, a, b, c):
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()] * 3)
        fn = builder.module.declare_intrinsic("llvm.fma", [ir.DoubleType()], fnty)
        return builder.call(fn, args)

    return sig, codegen


if USE_FMA:

    @nb.njit(inline="always")
    def fma(a, b, c):
        """``a*b + c`` with a single rounding."""
        return _fma_intrinsic(a, b, c)

else:

    @nb.njit(inline="always")
    def fma(a, b, c):
        return a * b + c
