"""Run the README tutorials with the Python port and compare with R.

Usage: python validation/tutorials.py <tutorial> [--out DIR] [--jobs N] [--compare-only]

Tutorials: t1_lung t1_lymphoma t2_crc t2_lymphoma t3_visium t4_bulk t5_scrna t6_presorted
R working copy (inputs, prebuilt models): $ECOTYPER_R_WORK (default work/ecotyper-r-src)
R reference text outputs: $ECOTYPER_R_REF (default work/r_ref, see README in validation/)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R_WORK = Path(os.environ.get("ECOTYPER_R_WORK", str(ROOT / "work" / "ecotyper-r-src")))
R_REF = Path(os.environ.get("ECOTYPER_R_REF", ROOT / "work" / "r_ref"))

# tutorial -> list of (python subdir, R reference subdir) pairs to compare
PAIRS = {
    "t1_lung": [("EcoTyper/Carcinoma/Carcinoma_Fractions/Cell_States/recovery/bulk_lung_data", "Carcinoma/states/bulk_lung_data"),
                ("EcoTyper/Carcinoma/Carcinoma_Fractions/Ecotypes/recovery/bulk_lung_data", "Carcinoma/ecotypes/bulk_lung_data"),
                ("RecoveryOutput/bulk_lung_data", "RecoveryOutput/bulk_lung_data")],
    "t1_lymphoma": [("EcoTyper/Lymphoma/Lymphoma_Fractions/Cell_States/recovery/bulk_lymphoma_data", "Lymphoma/states/bulk_lymphoma_data"),
                    ("EcoTyper/Lymphoma/Lymphoma_Fractions/Ecotypes/recovery/bulk_lymphoma_data", "Lymphoma/ecotypes/bulk_lymphoma_data"),
                    ("RecoveryOutput/bulk_lymphoma_data", "RecoveryOutput/bulk_lymphoma_data")],
    "t2_crc": [("EcoTyper/Carcinoma/Carcinoma_Fractions/Cell_States/recovery/scRNA/scRNA_CRC_data", "Carcinoma/states/scRNA/scRNA_CRC_data"),
               ("EcoTyper/Carcinoma/Carcinoma_Fractions/Ecotypes/recovery/scRNA_CRC_data", "Carcinoma/ecotypes/scRNA_CRC_data"),
               ("RecoveryOutput/scRNA_CRC_data", "RecoveryOutput/scRNA_CRC_data")],
    "t2_lymphoma": [("EcoTyper/Lymphoma/Lymphoma_Fractions/Cell_States/recovery/scRNA/scRNA_lymphoma_data", "Lymphoma/states/scRNA/scRNA_lymphoma_data"),
                    ("RecoveryOutput/scRNA_lymphoma_data", "RecoveryOutput/scRNA_lymphoma_data")],
    "t3_visium": [("EcoTyper/Carcinoma/Carcinoma_Fractions/Cell_States/recovery/VisiumBreast", "Carcinoma/states/VisiumBreast"),
                  ("VisiumOutput/VisiumBreast", "VisiumOutput/VisiumBreast")],
    "t6_presorted": [("EcoTyper/PresortedDiscovery/All_genes", "PresortedDiscovery/All_genes")],
    "t4_bulk": [("EcoTyper/MyDiscovery/Carcinoma_Fractions", "MyDiscovery/Carcinoma_Fractions")],
    "t5_scrna": [("EcoTyper/discovery_scRNA_CRC/Cell_type_specific_genes", "discovery_scRNA_CRC/Cell_type_specific_genes")],
}


def run(tut: str, out: Path, jobs: int, exact: bool = True, nmf_method: str = "brunet"):
    from ecotyper import workflows as wf

    ex = R_WORK / "example_data"
    models = R_WORK / "EcoTyper"
    # exact is passed only when False, so this script also drives older package versions
    kw = dict(n_jobs=jobs) if exact else dict(n_jobs=jobs, exact=False)
    dkw = dict(kw) if nmf_method == "brunet" else dict(kw, nmf_method=nmf_method)  # discovery only
    if tut == "t1_lung":
        wf.recover_bulk_files(ex / "bulk_lung_data.txt", ex / "bulk_lung_annotation.txt", "Carcinoma", out,
                              discovery_root=models, **kw)
    elif tut == "t1_lymphoma":
        wf.recover_bulk_files(ex / "bulk_lymphoma_data.txt", ex / "bulk_lymphoma_annotation.txt", "Lymphoma", out,
                              discovery_root=models, **kw)
    elif tut == "t2_crc":
        wf.recover_scrna_files(ex / "scRNA_CRC_data.txt", ex / "scRNA_CRC_annotation.txt", "Carcinoma", out,
                               discovery_root=models, **kw)
    elif tut == "t2_lymphoma":
        wf.recover_scrna_files(ex / "scRNA_lymphoma_data.txt", ex / "scRNA_lymphoma_annotation.txt", "Lymphoma", out,
                               discovery_root=models, **kw)
    elif tut == "t3_visium":
        wf.recover_visium_files(ex / "VisiumBreast", ex / "visium_fractions_example.txt", "Carcinoma", out,
                                recovery_name="VisiumBreast", discovery_root=models, **kw)
    elif tut == "t6_presorted":
        wf.discover_files(ex / "Tutorial_6" / "PresortedDiscovery", ex / "Tutorial_6" / "PresortedDiscovery_annotation.txt",
                          out, name="PresortedDiscovery", fractions="All_genes", scale_column="Histology", **dkw)
    elif tut == "t4_bulk":
        # CSx HiRes profiles: the Tutorial 6 inputs are the HiRes output of Tutorial 4
        wf.discover_files(ex / "Tutorial_6" / "PresortedDiscovery", ex / "bulk_lung_annotation.txt", out,
                          name="MyDiscovery", fractions="Carcinoma_Fractions", scale_column="Histology", **dkw)
    elif tut == "t5_scrna":
        from ecotyper.io import read_annotation, read_scrna_matrix

        res = wf.discover_scrna_frame(read_scrna_matrix(ex / "scRNA_CRC_data.txt"),
                                      read_annotation(ex / "scRNA_CRC_annotation.txt"),
                                      cophenetic_cutoff=0.975, **dkw)
        wf.write_discovery_scrna(res, out, "discovery_scRNA_CRC", write_large=True)
    else:
        raise SystemExit(f"unknown tutorial {tut}")


def compare(tut: str, out: Path) -> pd.DataFrame:
    from ecotyper.validate import compare_trees

    frames = []
    for py_sub, r_sub in PAIRS[tut]:
        d = compare_trees(out / py_sub, R_REF / r_sub)
        d.insert(0, "tree", r_sub)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tutorial")
    ap.add_argument("--out", default=str(ROOT / "work" / "py_out"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--no-compare", action="store_true", help="run only (used by benchmark.py)")
    ap.add_argument("--fast", action="store_true", help="fast mode (exact=False); report <tutorial>_fast.tsv")
    ap.add_argument("--nmf-method", default="brunet", help="discovery NMF solver; report <tutorial>_<method>.tsv")
    ap.add_argument("--report", default=None, help="report path (default validation/reports/<tutorial>[_fast][_<method>].tsv)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.compare_only:
        t = time.time()
        run(a.tutorial, out, a.jobs, exact=not a.fast, nmf_method=a.nmf_method)
        print(f"{a.tutorial}: python run {time.time() - t:.0f} s", file=sys.stderr)
    if a.no_compare:
        return
    df = compare(a.tutorial, out)
    rep = ROOT / "validation" / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    path = a.report or rep / f"{a.tutorial}{'_fast' if a.fast else ''}{'' if a.nmf_method == 'brunet' else '_' + a.nmf_method}.tsv"
    df.to_csv(path, sep="\t", index=False)
    n = len(df)
    print(f"{a.tutorial}: {n} files, {int(df['identical_bytes'].fillna(False).sum())} byte-identical, "
          f"{int(df['discrete_equal'].fillna(False).sum())} discrete-equal, missing {int(df['missing'].sum())}, "
          f"max rel {df['max_rel'].max():.3g}")


if __name__ == "__main__":
    main()
