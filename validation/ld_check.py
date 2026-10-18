"""Bit-compare the port's R-compat reductions against R itself.

    python validation/ld_check.py gen DIR      # writes DIR/X.bin
    Rscript validation/ld_check.R DIR          # writes DIR/*.R.bin
    python validation/ld_check.py cmp DIR      # compares

On x86-64 this exercises the long-double kernels (rcompat/_ld.c).
"""
import sys
from pathlib import Path

import numpy as np

NR, NC = 300, 40


def gen(d):
    rng = np.random.default_rng(1)
    X = rng.lognormal(0.0, 3.0, (NR, NC)) * rng.choice([1e-3, 1.0, 1e4], (NR, NC))
    X[:, 5] = np.round(X[:, 5])  # ties / exact values
    X[:, 7] = 3.25  # constant column (sd == 0)
    d.mkdir(parents=True, exist_ok=True)
    np.asfortranarray(X).T.copy().T.tofile(d / "X.bin")  # column-major
    (d / "dim.txt").write_text(f"{NR} {NC}\n")
    return X


def cmp(d):
    from ecotyper.rcompat import stats as S

    X = np.fromfile(d / "X.bin").reshape((NC, NR)).T.copy()
    ld = S._LD is not None
    got = {
        "sum": np.array([S.r_sum(X[:, j]) for j in range(NC)]),
        "mean": np.array([S.r_mean(X[:, j]) for j in range(NC)]),
        "colMeans": S.col_means(X),
        "rowMeans": S.row_means(X),
        "rowMean_apply": S.apply_mean_rows(X),
        "var": np.array([S.r_var(X[:, j]) for j in range(NC)]),
        "rowvar": S.row_var(X),
        "cor": S.cor_matrix(X),
        "cor2": np.array([S.cor_vectors(X[:, j], X[:, j + 1]) for j in range(NC - 1)]),
        "scale_rows": S.scale_rows(X),
        "scale_vec": np.stack([S.scale_vec(X[:, j]) for j in range(NC)], axis=1),
        "colnorm": S.col_normalize(X),
    }
    shapes = {"cor": (NC, NC), "scale_rows": (NR, NC), "scale_vec": (NR, NC), "colnorm": (NR, NC)}
    bad = 0
    print(f"long-double kernels: {'ON' if ld else 'off'}")
    for k, v in got.items():
        r = np.fromfile(d / f"{k}.R.bin")
        if k in shapes:  # R writes column-major
            r = r.reshape(shapes[k][::-1]).T
        r = r.reshape(v.shape)
        same = np.array_equal(v, r) or np.array_equal(np.isnan(v), np.isnan(r)) and np.array_equal(v[~np.isnan(v)], r[~np.isnan(r)])
        nd = int((v.view(np.int64) != r.view(np.int64)).sum()) if not same else 0
        mx = np.nanmax(np.abs(v - r) / np.maximum(np.abs(r), 1e-300)) if not same else 0.0
        bad += 0 if same else 1
        print(f"{k:14s} {'IDENTICAL' if same else f'DIFF n={nd}/{v.size} maxrel={mx:.2e}'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    mode, d = sys.argv[1], Path(sys.argv[2])
    gen(d) if mode == "gen" else cmp(d)
