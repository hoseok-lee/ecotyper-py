"""Agreement of a discovery run with the R reference: chosen ranks and adjusted Rand index (ARI)
of the cell state and ecotype assignments. Used for the alternative NMF solver
(``--nmf-method cd``), where outputs are not expected to be equal to R's.

Usage: python validation/solver_agreement.py PY_DIR R_DIR [--label NAME] [--out TSV]

PY_DIR and R_DIR are ``<discovery name>/<fractions>`` folders (with ``Cell_States/discovery`` and
``Ecotypes/discovery``). Samples missing from one side's assignment (filtered states,
unassigned ecotypes) are labelled ``unassigned`` before computing the ARI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _chosen(ct_dir: Path):
    for d in sorted(ct_dir.iterdir()):
        if d.is_dir() and (d / "initial_state_assignment.txt").exists():
            return d
    return None


def _labels(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    return df.set_index(df["ID"].astype(str))[col].astype(str)


def _ari(a: pd.Series, b: pd.Series, ids=None) -> tuple[float, int]:
    ids = sorted(set(a.index) | set(b.index)) if ids is None else ids
    x = a.reindex(ids).fillna("unassigned")
    y = b.reindex(ids).fillna("unassigned")
    return adjusted_rand_score(x, y), len(ids)


def agreement(py: Path, r: Path) -> pd.DataFrame:
    rows = []
    cs_py, cs_r = py / "Cell_States" / "discovery", r / "Cell_States" / "discovery"
    for ct_dir in sorted(p for p in cs_r.iterdir() if p.is_dir()):
        dr, dp = _chosen(ct_dir), _chosen(cs_py / ct_dir.name) if (cs_py / ct_dir.name).exists() else None
        row = {"cell_type": ct_dir.name, "rank_r": int(dr.name) if dr else None,
               "rank_py": int(dp.name) if dp else None}
        if dr and dp:
            ia_r = _labels(dr / "initial_state_assignment.txt", "State")
            row["ari_initial"], row["n"] = _ari(_labels(dp / "initial_state_assignment.txt", "State"), ia_r,
                                                list(ia_r.index))
            if (dr / "state_assignment.txt").exists() and (dp / "state_assignment.txt").exists():
                row["ari_states"], _ = _ari(_labels(dp / "state_assignment.txt", "State"),
                                            _labels(dr / "state_assignment.txt", "State"), list(ia_r.index))
        rows.append(row)
    er, ep = r / "Ecotypes" / "discovery" / "ecotype_assignment.txt", py / "Ecotypes" / "discovery" / "ecotype_assignment.txt"
    if er.exists() and ep.exists():
        ari, n = _ari(_labels(ep, "Ecotype"), _labels(er, "Ecotype"))
        rows.append({"cell_type": "(ecotypes)", "ari_states": ari, "n": n,
                     "rank_r": pd.read_csv(r / "Ecotypes" / "discovery" / "ecotypes.txt", sep="\t")["Ecotype"].nunique(),
                     "rank_py": pd.read_csv(py / "Ecotypes" / "discovery" / "ecotypes.txt", sep="\t")["Ecotype"].nunique()})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("py")
    ap.add_argument("r")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = agreement(Path(a.py), Path(a.r))
    df.insert(0, "run", a.label)
    if a.out:
        df.to_csv(a.out, sep="\t", index=False)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
