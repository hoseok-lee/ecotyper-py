"""High-level workflows mirroring the R wrapper scripts.

* :func:`recover_bulk` ↔ ``EcoTyper_recovery_bulk.R``
* :func:`recover_scrna` ↔ ``EcoTyper_recovery_scRNA.R``
* :func:`recover_visium` ↔ ``EcoTyper_recovery_visium.R``
* :func:`discover_bulk`, :func:`discover_presorted`, :func:`discover_scrna`
  ↔ the discovery wrappers (see :mod:`ecotyper.discovery`).

AnnData conventions (``key = "ecotyper"`` by default):

``adata.obs[f"{key}_state_{cell_type}"]``
    assigned cell state (``S01``…; NaN if the sample's dominant initial state
    was filtered out during discovery);
``adata.obsm[f"{key}_abundance_{cell_type}"]``
    cell state abundances (samples x states, columns ``S01``…);
``adata.obs[f"{key}_ecotype"]``, ``adata.obs[f"{key}_max_ecotype"]``,
``adata.obs[f"{key}_ecotype_q"]``
    ecotype assignment (``Unassigned`` if not significant), most abundant
    ecotype and its BH-adjusted p-value (bulk/Visium);
``adata.obsm[f"{key}_ecotype_abundance"]``
    ecotype abundances (samples x ecotypes);
``adata.uns[key]``
    run parameters, number of NMF iterations per cell type, and the full
    per-step result objects under ``"results"``.

R starts each per-cell-type script in a fresh process. The R reference runs
seed every process with ``set.seed(seed)`` (R itself leaves NMFpredict
unseeded); the port reseeds ``RRandom(seed)`` for every cell type the same way.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import recovery as rec
from .io import (annotation_from_anndata, from_anndata, read_annotation, read_bulk_matrix,
                 read_fractions, read_scrna_matrix)
from .model import EcoTyperModel
from .rcompat.rng import RRandom
from .rlayout import write_ecotype_recovery, write_recovery_output, write_state_recovery


def _model(model, discovery_root=None) -> EcoTyperModel:
    if isinstance(model, EcoTyperModel):
        return model
    if discovery_root is not None:
        return EcoTyperModel.load(discovery_root, model)
    return EcoTyperModel.load(model)


def _run_parallel(fn, tasks, n_jobs):
    if n_jobs is None or n_jobs <= 1 or len(tasks) <= 1:
        return [fn(*t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(fn, *t) for t in tasks]
        return [f.result() for f in futs]


def _bulk_state_task(expr, ctm, seed, annotation, kind):
    return rec.recover_states(expr, ctm, RRandom(seed), annotation, kind=kind)


def _store(adata, key, model, states, eco, seed, kind):
    if adata is None:
        return
    for ct, r in states.items():
        ab = r.abundances.T.reindex(adata.obs_names)
        ab.columns = [str(c) for c in ab.columns]
        adata.obsm[f"{key}_abundance_{ct}"] = ab
        st = pd.Series(np.nan, index=adata.obs_names, dtype=object)
        st.loc[r.assignment["ID"].to_numpy()] = r.assignment["State"].to_numpy()
        adata.obs[f"{key}_state_{ct}"] = pd.Categorical(st, categories=list(r.abundances.index))
    if eco is not None and kind != "scRNA":
        ia = eco.initial_assignment.set_index("ID")
        adata.obs[f"{key}_ecotype"] = pd.Categorical(ia["Ecotype"].reindex(adata.obs_names),
                                                     categories=list(eco.ecotype_abundance.index) + ["Unassigned"])
        adata.obs[f"{key}_max_ecotype"] = ia["MaxEcotype"].reindex(adata.obs_names).to_numpy()
        adata.obs[f"{key}_ecotype_q"] = ia["AssignmentQ"].reindex(adata.obs_names).to_numpy(dtype=float)
        adata.obsm[f"{key}_ecotype_abundance"] = eco.ecotype_abundance.T.reindex(adata.obs_names)
    adata.uns[key] = {
        "discovery": model.name, "fractions": model.fractions, "seed": seed, "kind": kind,
        "niter": {ct: int(r.niter) for ct, r in states.items()},
    }


# ----------------------------------------------------------------------------
# bulk
# ----------------------------------------------------------------------------
def recover_bulk_frame(expr: pd.DataFrame, model: EcoTyperModel, annotation: pd.DataFrame | None = None, *,
                       seed: int = 42, n_jobs: int = 1, r_exact: bool = True, kind: str = "bulk"):
    """Bulk recovery on a genes x samples DataFrame (values as read by R).

    Returns ``(state_results, ecotype_result)``."""
    data = rec.prepare_bulk_expression(expr)
    clin = rec.read_clinical(list(data.columns), annotation)
    cts = model.cell_type_order
    res = _run_parallel(_bulk_state_task, [(data, model.cell_types[ct], seed, clin, kind) for ct in cts], n_jobs)
    states = dict(zip(cts, res))
    eco = None
    if model.ecotypes is not None:
        eco = rec.recover_ecotypes(states, model.ecotypes, cts, annotation, r_exact=r_exact)
    return states, eco


def recover_bulk(adata, model, *, discovery_root=None, layer=None, seed: int = 42, n_jobs: int = 1,
                 r_exact: bool = True, key: str = "ecotyper"):
    """Recover cell states and ecotypes in bulk samples (``adata``: samples x genes).

    ``model`` is an :class:`EcoTyperModel`, a path to a model folder, or a
    discovery name (``"Carcinoma"``, ``"Lymphoma"``…) with ``discovery_root``
    pointing to the ``EcoTyper`` folder."""
    model = _model(model, discovery_root)
    expr = from_anndata(adata, layer)
    ann = annotation_from_anndata(adata)
    states, eco = recover_bulk_frame(expr, model, ann, seed=seed, n_jobs=n_jobs, r_exact=r_exact)
    _store(adata, key, model, states, eco, seed, "bulk")
    adata.uns[key]["results"] = {"states": states, "ecotypes": eco}
    return states, eco


def recover_bulk_files(matrix, annotation, model, out_dir, *, discovery_root=None, recovery_name=None,
                       seed: int = 42, n_jobs: int = 1, r_exact: bool = True):
    """File-based equivalent of ``Rscript EcoTyper_recovery_bulk.R -m matrix -a annotation``.

    Writes the intermediate ``EcoTyper/<discovery>/<fractions>/{Cell_States,Ecotypes}/recovery``
    tree under ``out_dir/EcoTyper`` and the user-facing folder
    ``out_dir/RecoveryOutput/<recovery_name>``."""
    model = _model(model, discovery_root)
    if recovery_name is None:
        recovery_name = Path(matrix).name
        for suf in (".txt", ".tsv"):
            if recovery_name.endswith(suf):
                recovery_name = recovery_name[: -len(suf)]
    expr = read_bulk_matrix(matrix)
    ann = read_annotation(annotation)
    states, eco = recover_bulk_frame(expr, model, ann, seed=seed, n_jobs=n_jobs, r_exact=r_exact)
    out = Path(out_dir)
    base = out / "EcoTyper" / model.name / model.fractions
    for ct, r in states.items():
        write_state_recovery(r, base / "Cell_States" / "recovery" / recovery_name / ct / str(model.cell_types[ct].n_states))
    if eco is not None:
        write_ecotype_recovery(eco, base / "Ecotypes" / "recovery" / recovery_name)
    write_recovery_output(states, eco, out / "RecoveryOutput" / recovery_name, model.cell_type_order)
    return states, eco


# ----------------------------------------------------------------------------
# scRNA-seq
# ----------------------------------------------------------------------------
def _scrna_state_task(expr_all, annotation, ctm, seed, subsample):
    """``state_recovery_scRNA.R`` for one cell type (``None`` if no cells)."""
    ct = ctm.cell_type
    ann = annotation[annotation["CellType"].astype(str) == ct]
    if len(ann) == 0:
        return None
    cols = set(expr_all.columns)
    ann = ann[[str(i) in cols for i in ann["ID"]]]
    if len(ann) == 0:
        raise ValueError(f"The cell IDs do not match between the expression matrix and the annotation for '{ct}'")
    ids = [str(i) for i in ann["ID"]]
    expr = expr_all[ids]
    X = expr.to_numpy(dtype=np.float64)
    if not (X >= 0).all():
        raise ValueError("Not all values in the expression matrix are positive. Could not take log2!")
    from .rcompat.stats import log2p1

    expr = pd.DataFrame(log2p1(X), index=expr.index, columns=ids)
    ann = ann.copy()
    ann.index = pd.Index(ids, dtype=object)
    rng = RRandom(seed)
    data = rec.align_rows(expr, [str(g) for g in ctm.initial_gene_info["Gene"]])
    if subsample >= 50 and data.shape[1] > subsample:
        rng.set_seed(1234)
        pick = rng.sample_int(data.shape[1], subsample) - 1
        data = data.iloc[:, pick]
        ann = ann.iloc[[ids.index(c) for c in data.columns]]
    return rec.recover_states(data, ctm, rng, ann, kind="scRNA")


def recover_scrna_frame(expr_all: pd.DataFrame, annotation: pd.DataFrame, model: EcoTyperModel, *,
                        seed: int = 42, subsample: int = -1, n_jobs: int = 1):
    cts = model.cell_type_order
    res = _run_parallel(_scrna_state_task, [(expr_all, annotation, model.cell_types[ct], seed, subsample)
                                            for ct in cts], n_jobs)
    states = {ct: r for ct, r in zip(cts, res) if r is not None}
    eco = None
    if model.ecotypes is not None and "Sample" in annotation.columns:
        eco = rec.recover_ecotypes_scrna(states, model, annotation)
    return states, eco


def recover_scrna(adata, model, *, discovery_root=None, layer=None, seed: int = 42, subsample: int = -1,
                  n_jobs: int = 1, key: str = "ecotyper"):
    """Recover cell states in single cells (``adata``: cells x genes, raw-scale
    expression; ``obs`` must have ``CellType`` and, for ecotypes, ``Sample``).

    Ecotype abundances are per sample, returned (not stored in ``adata``)."""
    model = _model(model, discovery_root)
    expr = from_anndata(adata, layer)
    ann = annotation_from_anndata(adata)
    states, eco = recover_scrna_frame(expr, ann, model, seed=seed, subsample=subsample, n_jobs=n_jobs)
    _store(adata, key, model, states, None, seed, "scRNA")
    adata.uns[key]["results"] = {"states": states, "ecotypes": eco}
    return states, eco


def recover_scrna_files(matrix, annotation, model, out_dir, *, discovery_root=None, recovery_name=None,
                        seed: int = 42, subsample: int = -1, n_jobs: int = 1):
    """File-based equivalent of ``Rscript EcoTyper_recovery_scRNA.R``."""
    model = _model(model, discovery_root)
    if recovery_name is None:
        recovery_name = Path(matrix).name
        for suf in (".txt", ".tsv"):
            if recovery_name.endswith(suf):
                recovery_name = recovery_name[: -len(suf)]
    expr = read_scrna_matrix(matrix)
    ann = read_annotation(annotation)
    states, eco = recover_scrna_frame(expr, ann, model, seed=seed, subsample=subsample, n_jobs=n_jobs)
    out = Path(out_dir)
    base = out / "EcoTyper" / model.name / model.fractions
    for ct, r in states.items():
        write_state_recovery(r, base / "Cell_States" / "recovery" / "scRNA" / recovery_name / ct /
                             str(model.cell_types[ct].n_states))
    if eco is not None:
        write_ecotype_recovery(eco, base / "Ecotypes" / "recovery" / recovery_name)
    write_recovery_output(states, eco, out / "RecoveryOutput" / recovery_name, model.cell_type_order)
    return states, eco


# ----------------------------------------------------------------------------
# Visium
# ----------------------------------------------------------------------------
def read_visium(matrix_dir):
    """``spatial_load_visium_data.R``: 10x Visium folder → (genes x spots counts,
    spot annotation with ``ID``, ``X`` (array row), ``Y`` (array col))."""
    import gzip

    from scipy.io import mmread

    from .rcompat.frame import make_names, r_match

    d = Path(matrix_dir)
    with gzip.open(d / "matrix.mtx.gz") as fh:
        mat = mmread(fh).tocsc()
    with gzip.open(d / "features.tsv.gz", "rt") as fh:
        feats = [line.rstrip("\n").split("\t") for line in fh if line.strip()]
    with gzip.open(d / "barcodes.tsv.gz", "rt") as fh:
        bcs = [line.rstrip("\n").split("\t")[0] for line in fh if line.strip()]
    genes = [f[1] for f in feats]
    keep = []
    seen = set()
    for i, g in enumerate(genes):
        if g not in seen:
            seen.add(g)
            keep.append(i)
    X = mat[keep].toarray().astype(np.float64)
    pos = pd.read_csv(d / "tissue_positions_list.csv", header=None, dtype={0: str})
    pos_ids = make_names(list(pos[0]))
    cols = make_names(bcs)
    idx = r_match(cols, pos_ids)
    expr = pd.DataFrame(X, index=pd.Index([genes[i] for i in keep], dtype=object),
                        columns=[pos_ids[i] for i in idx])
    ann = pd.DataFrame({"ID": list(expr.columns),
                        "X": pos[2].to_numpy(dtype=float)[idx],
                        "Y": pos[3].to_numpy(dtype=float)[idx]})
    return expr, ann


def recover_visium_frame(expr, spots, model: EcoTyperModel, fractions: pd.DataFrame, *, sample_name: str,
                         malignant_cell: str | None = "Epithelial.cells", seed: int = 42, n_jobs: int = 1,
                         r_exact: bool = True):
    """Returns ``(state_results, state_abundances_long, state_abundances, ecotype_abundances)``."""
    states, _ = recover_bulk_frame(expr, model, spots, seed=seed, n_jobs=n_jobs, r_exact=r_exact, kind="visium")
    # model.ecotypes is not needed for state recovery; bulk ecotype recovery is not run for Visium
    cts = model.cell_type_order
    long, wide = rec.spatial_states(states, cts, fractions, spots, model.cell_types, malignant_cell, sample_name,
                                    r_exact=r_exact)
    eco = rec.spatial_ecotypes(wide, model.ecotypes, r_exact=r_exact)
    return states, long, wide, eco


def recover_visium_files(visium_dir, fractions_path, model, out_dir, *, recovery_name, discovery_root=None,
                         malignant_cell: str | None = "Epithelial.cells", seed: int = 42, n_jobs: int = 1,
                         r_exact: bool = True):
    """File-based equivalent of ``Rscript EcoTyper_recovery_visium.R -c config.yml``
    with precomputed fractions."""
    from .rcompat.io import write_table

    model = _model(model, discovery_root)
    expr, spots = read_visium(visium_dir)
    frac = read_fractions(fractions_path)
    states, long, wide, eco = recover_visium_frame(expr, spots, model, frac, sample_name=recovery_name,
                                                   malignant_cell=malignant_cell, seed=seed, n_jobs=n_jobs,
                                                   r_exact=r_exact)
    out = Path(out_dir)
    base = out / "EcoTyper" / model.name / model.fractions / "Cell_States" / "recovery" / recovery_name
    for ct, r in states.items():
        r.heatmap_data = None  # Visium does not write heatmap tables
        r.heatmap_top_ann = None
        write_state_recovery(r, base / ct / str(model.cell_types[ct].n_states))
    write_table(long, base / "state_abundances_long.txt", row_names=False)
    write_table(wide, base / "state_abundances.txt", row_names=False)
    write_table(eco, base / "ecotype_abundances.txt")
    vo = out / "VisiumOutput" / recovery_name
    write_table(wide, vo / "state_abundances.txt", row_names=False)
    write_table(eco, vo / "ecotype_abundances.txt")
    return states, long, wide, eco
