"""Build the per-file tables of VALIDATION.md from validation/reports/<tutorial>.tsv.

Usage: python validation/make_validation_md.py > validation/tables.md"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REP = Path(__file__).resolve().parent / "reports"
ORDER = ["t1_lung", "t1_lymphoma", "t2_crc", "t2_lymphoma", "t3_visium", "t6_presorted", "t4_bulk", "t5_scrna"]


def fmt(x):
    if pd.isna(x):
        return "–"
    return "0" if x == 0 else f"{x:.2g}"


def main():
    for tut in ORDER:
        p = REP / f"{tut}.tsv"
        if not p.exists():
            print(f"### {tut}\n\nnot run\n")
            continue
        df = pd.read_csv(p, sep="\t")
        for c in ("missing", "identical_bytes", "shape_equal", "discrete_equal"):
            df[c] = df[c].fillna(False).astype(bool)
        n = len(df)
        print(f"### {tut}\n")
        print(f"{n} files; {df.identical_bytes.sum()} byte-identical; {df.discrete_equal.sum()} discrete-equal; "
              f"{df.missing.sum()} missing; max abs {fmt(df.max_abs.max())}; max rel {fmt(df.max_rel.max())}.\n")
        print("| file | status | numeric cells | differing | max abs | max rel |")
        print("|---|---|---:|---:|---:|---:|")
        for _, r in df.iterrows():
            if r.missing:
                st = "MISSING"
            elif r.identical_bytes:
                st = "identical"
            elif not r.shape_equal:
                st = "SHAPE DIFFERS"
            elif not r.discrete_equal:
                st = "DISCRETE DIFFERS"
            else:
                st = "numeric diff"
            n_num = "" if pd.isna(r.get("n_numeric")) else int(r.n_numeric)
            n_d = "" if pd.isna(r.get("n_diff_numeric")) else int(r.n_diff_numeric)
            print(f"| {r.tree}/{r.file} | {st} | {n_num} | {n_d} | {fmt(r.get('max_abs'))} | {fmt(r.get('max_rel'))} |")
        print()


if __name__ == "__main__":
    main()
