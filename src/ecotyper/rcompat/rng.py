"""R's default random number generator (Mersenne-Twister + Rejection sampling).

This reproduces, draw for draw, ``set.seed()``, ``unif_rand()``, ``runif()`` and
``sample()``/``sample.int()`` of R >= 3.6 with the default
``RNGkind("Mersenne-Twister", "Inversion", "Rejection")`` (see R's
``src/main/RNG.c`` and ``src/main/random.c``).
"""
from __future__ import annotations

import os

import numba as nb
import numpy as np

_N = 624
_M = 397
_I2_32M1 = 2.328306437080797e-10  # 1/(2^32 - 1)


@nb.njit(cache=True)
def _lcg(seed):
    return (69069 * seed + 1) & 0xFFFFFFFF


@nb.njit(cache=True)
def _set_seed(state, seed):
    """R's RNG_Init for Mersenne-Twister. ``state[0]`` is ``mti`` (dummy[0])."""
    s = seed & 0xFFFFFFFF
    for _ in range(50):
        s = _lcg(s)
    for j in range(_N + 1):
        s = _lcg(s)
        state[j] = s
    state[0] = _N  # FixupSeeds(initial=1): dummy[0] = 624


@nb.njit(cache=True)
def _mt_genrand(state):
    mti = state[0]
    if mti >= _N:
        # (mti == N+1 cannot happen after set_seed)
        for kk in range(_N - _M):
            y = (state[1 + kk] & 0x80000000) | (state[1 + kk + 1] & 0x7FFFFFFF)
            state[1 + kk] = state[1 + kk + _M] ^ (y >> 1) ^ (0x9908B0DF if (y & 1) else 0)
        for kk in range(_N - _M, _N - 1):
            y = (state[1 + kk] & 0x80000000) | (state[1 + kk + 1] & 0x7FFFFFFF)
            state[1 + kk] = state[1 + kk + (_M - _N)] ^ (y >> 1) ^ (0x9908B0DF if (y & 1) else 0)
        y = (state[_N] & 0x80000000) | (state[1] & 0x7FFFFFFF)
        state[_N] = state[_M] ^ (y >> 1) ^ (0x9908B0DF if (y & 1) else 0)
        mti = 0
    y = state[1 + mti]
    mti += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y &= 0xFFFFFFFF
    y ^= y >> 18
    state[0] = mti
    return float(y) * 2.3283064365386963e-10


@nb.njit(cache=True)
def _unif_rand(state):
    x = _mt_genrand(state)
    if x <= 0.0:
        return 0.5 * _I2_32M1
    if (1.0 - x) <= 0.0:
        return 1.0 - 0.5 * _I2_32M1
    return x


@nb.njit(cache=True)
def _runif(state, n, a, b, out):
    if a == b:
        for i in range(n):
            out[i] = a
        return
    for i in range(n):
        u = _unif_rand(state)
        while u <= 0.0 or u >= 1.0:
            u = _unif_rand(state)
        out[i] = a + (b - a) * u


@nb.njit(cache=True)
def _rbits(state, bits):
    v = np.int64(0)
    n = 0
    while n <= bits:
        v1 = np.int64(np.floor(_unif_rand(state) * 65536))
        v = 65536 * v + v1
        n += 16
    return float(v & ((np.int64(1) << bits) - 1))


@nb.njit(cache=True)
def _unif_index(state, dn):
    if dn <= 0:
        return 0.0
    bits = int(np.ceil(np.log2(dn)))
    dv = _rbits(state, bits)
    while dn <= dv:
        dv = _rbits(state, bits)
    return dv


@nb.njit(cache=True)
def _sample_int(state, n, k, replace, out):
    """``sample.int(n, k, replace)`` — 1-based results written to ``out``."""
    if replace or k < 2:
        for i in range(k):
            out[i] = int(_unif_index(state, float(n))) + 1
        return
    x = np.empty(n, dtype=np.int64)
    for i in range(n):
        x[i] = i
    nn = n
    for i in range(k):
        j = int(_unif_index(state, float(nn)))
        out[i] = x[j] + 1
        nn -= 1
        x[j] = x[nn]


@nb.njit(cache=True)
def _permute_rows(state, data, out):
    """``t(apply(data, 1, sample))`` for a numeric matrix with >1 column."""
    nr, nc = data.shape
    idx = np.empty(nc, dtype=np.int64)
    for r in range(nr):
        _sample_int(state, nc, nc, False, idx)
        for c in range(nc):
            out[r, c] = data[r, idx[c] - 1]


class RRandom:
    """Stateful emulation of R's RNG (``.Random.seed``)."""

    def __init__(self, seed: int | None = None):
        self.state = np.zeros(_N + 1, dtype=np.int64)
        if seed is None:
            # R seeds an unseeded session from time and PID; use OS entropy instead
            seed = int.from_bytes(os.urandom(4), "little") - 2**31
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """``set.seed(seed)`` (``seed`` is coerced like ``as.integer``)."""
        _set_seed(self.state, int(seed))

    def copy(self) -> "RRandom":
        other = RRandom.__new__(RRandom)
        other.state = self.state.copy()
        return other

    @property
    def random_seed(self) -> np.ndarray:
        """``.Random.seed`` as R shows it (first element 10403 = kind code)."""
        s = self.state.astype(np.uint32).view(np.int32).astype(np.int64)
        return np.concatenate([[10403], s])

    def unif_rand(self, n: int = 1) -> np.ndarray:
        out = np.empty(n)
        _runif(self.state, n, 0.0, 1.0, out)
        return out

    def runif(self, n: int, min: float = 0.0, max: float = 1.0) -> np.ndarray:
        out = np.empty(int(n))
        _runif(self.state, int(n), float(min), float(max), out)
        return out

    def sample_int(self, n: int, size: int | None = None, replace: bool = False) -> np.ndarray:
        """``sample.int(n, size, replace)``; returns 1-based integers like R."""
        if size is None:
            size = n
        if not replace and size > n:
            raise ValueError("cannot take a sample larger than the population when 'replace = FALSE'")
        out = np.empty(int(size), dtype=np.int64)
        _sample_int(self.state, int(n), int(size), bool(replace), out)
        return out

    def sample(self, x, size: int | None = None, replace: bool = False):
        """``sample(x, size, replace)`` for a vector ``x`` of length > 1."""
        x = np.asarray(x)
        if x.ndim == 0 or len(x) == 1:
            raise ValueError("R's sample() treats a length-1 numeric x as 1:x; use sample_int")
        idx = self.sample_int(len(x), len(x) if size is None else size, replace)
        return x[idx - 1]

    def permute_rows(self, data: np.ndarray) -> np.ndarray:
        """``t(apply(data, 1, sample))`` (R's bootstrap in calculate_recovery_significance)."""
        data = np.ascontiguousarray(data, dtype=np.float64)
        out = np.empty_like(data)
        _permute_rows(self.state, data, out)
        return out
