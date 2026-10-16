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
from .rlayout import _wm, write_ecotype_recovery, write_recovery_output, write_state_recovery


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
    from .rcompat.frame import make_names

    fr = frac.copy()
    fr.columns = make_names(list(fr.columns))
    write_table(fr, base / f"{recovery_name}_CSx_fractions.txt", row_names=False)
    write_table(long, base / "state_abundances_long.txt", row_names=False)
    eco_long = ecotype_abundances_long(eco)
    write_table(eco_long, base / "ecotype_abundances_long.txt")
    write_table(wide, base / "state_abundances.txt", row_names=False)
    write_table(eco, base / "ecotype_abundances.txt")
    vo = out / "VisiumOutput" / recovery_name
    write_table(wide, vo / "state_abundances.txt", row_names=False)
    write_table(eco, vo / "ecotype_abundances.txt")
    return states, long, wide, eco


def ecotype_abundances_long(eco: pd.DataFrame) -> pd.DataFrame:
    """``melt(ecotype_data, id.vars = c("ID", "X", "Y", "Sample", "Malignant"))``."""
    ids = ["ID", "X", "Y", "Sample", "Malignant"]
    parts = []
    for c in [c for c in eco.columns if c not in ids]:
        d = eco[ids].copy().reset_index(drop=True)
        d["Ecotype"] = c
        d["Abundance"] = eco[c].to_numpy()
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out.index = pd.Index([str(i + 1) for i in range(len(out))], dtype=object)
    return out


# ----------------------------------------------------------------------------
# discovery (bulk downstream of CIBERSORTx, presorted)
# ----------------------------------------------------------------------------
def _nmf_rank_task(V, rank, restarts):
    from . import discovery as dsc

    fits = [dsc.run_nmf(V, rank, r) for r in range(1, restarts + 1)]
    merged, metrics = dsc.rank_metrics(fits, rank)
    return merged, metrics, [int(f.niter) for f in fits]


class DiscoveryResult(dict):
    """Results of a discovery run (attribute access to the dict keys)."""

    __getattr__ = dict.__getitem__


def discover_from_profiles(profiles: dict, annotation: pd.DataFrame | None = None, *, scale_column=None,
                           max_states: int = 20, restarts: int = 5, cophenetic_cutoff: float = 0.95,
                           min_states: int = 3, n_jobs: int = 1, r_exact: bool = True, verbose: bool = True):
    """Cell state + ecotype discovery from per-cell-type expression profiles
    (``EcoTyper_discovery_bulk.R`` steps 3–6 and 8 with CSx HiRes profiles as
    input, or ``EcoTyper_discovery_presorted.R`` with "no filter").

    ``profiles``: cell type → genes x samples DataFrame (values as read by
    ``read.delim(row.names = 1)``), in the order of ``classes.txt``."""
    from . import discovery as dsc

    feats = {}
    for ct, raw in profiles.items():
        f = dsc.extract_features(raw, ct, annotation, scale_column)
        if f is None:
            if verbose:
                print(f"Only {raw.shape[0]} genes are available for '{ct}'; skipping (need >= 50).")
            continue
        feats[ct] = f
    tasks, keys = [], []
    for ct, f in feats.items():
        V = dsc.nmf_input(f.top_scaled, r_exact)
        for k in range(2, max_states + 1):
            tasks.append((V, k, restarts))
            keys.append((ct, k))
    res = _run_parallel(_nmf_rank_task, tasks, n_jobs)
    merged = {key: r[0] for key, r in zip(keys, res)}
    metrics = {key: r[1] for key, r in zip(keys, res)}
    niter = {key: r[2] for key, r in zip(keys, res)}
    rank_tables = {ct: pd.DataFrame([metrics[(ct, k)] for k in range(2, max_states + 1)]) for ct in feats}
    ranks = dsc.select_ranks(rank_tables, max_states, cophenetic_cutoff, r_exact)
    initial, states = {}, {}
    cts = [str(c) for c in ranks["CellType"]]
    for ct, k in zip(cts, ranks["Chosen_Rank"]):
        k = int(k)
        initial[ct] = dsc.initial_states(merged[(ct, k)], feats[ct].top_scaled, ct, k, r_exact)
        states[ct] = dsc.first_filter(initial[ct], feats[ct], annotation, r_exact)
    eco = dsc.discover_ecotypes(initial, states, cts, min_states)
    assigned = dsc.assign_samples(states, eco.ecotypes, cts, annotation, r_exact)
    return DiscoveryResult(features=feats, merged=merged, rank_tables=rank_tables, niter=niter, ranks=ranks,
                           initial=initial, states=states, ecotypes=eco, assignment=assigned,
                           params=dict(max_states=max_states, restarts=restarts,
                                       cophenetic_cutoff=cophenetic_cutoff, min_states=min_states))


def read_profiles(input_dir, classes=None) -> dict:
    """Per-cell-type profiles ``<input_dir>/<cell type>.txt`` (``read.delim(row.names = 1)``).
    ``classes``: cell type order (default: ``list.files`` order)."""
    d = Path(input_dir)
    if classes is None:
        if (d / "classes.txt").exists():
            classes = [c.strip('"') for c in (d / "classes.txt").read_text().strip().split("\t")]
        else:
            classes = sorted(p.name[:-4] for p in d.glob("*.txt") if p.name != "classes.txt")
    return {ct: read_bulk_matrix(d / f"{ct}.txt") for ct in classes}


def write_discovery(res: DiscoveryResult, out_dir, name: str, fractions: str) -> Path:
    """Write the R layout ``<out_dir>/EcoTyper/<name>/<fractions>/...``."""
    from .rcompat.io import write_table

    base = Path(out_dir) / "EcoTyper" / name / fractions
    cs = base / "Cell_States" / "discovery"
    p = res.params
    for ct, f in res.features.items():
        d = cs / ct
        d.mkdir(parents=True, exist_ok=True)
        (d / "expression_top_genes_log2.txt").write_text('"x"\n')  # R writes utils::data here (sic)
        _wm(f.top_log2, d / "expression_full_matrix_log2.txt")
        _wm(f.top_scaled, d / "expression_top_genes_scaled.txt")
        _wm(f.full_scaled, d / "expression_full_matrix_scaled.txt")
        for k in range(2, p["max_states"] + 1):
            row = res.rank_tables[ct].iloc[[k - 2]].reset_index(drop=True)
            row["n_clusters"] = row["n_clusters"].astype(int)
            write_table(row, d / str(k) / "rank_data.txt", row_names=False)
    rs = base / "Analysis" / "rank_selection"
    write_table(res.ranks, rs / "rank_data.txt", row_names=False)
    from .rcompat.io import format_real

    write_table(res.ranks, rs / f"rank_data_cutoff_{format_real(p['cophenetic_cutoff'])}.txt", row_names=False)
    for ct, ini in res.initial.items():
        d = cs / ct / str(ini.rank)
        _wm(ini.W, d / "W.txt")
        _wm(ini.H, d / "H.txt")
        _wm(ini.initial_abundances, d / "initial_state_abundances.txt")
        write_table(ini.initial_assignment, d / "initial_state_assignment.txt", row_names=False)
        write_table(ini.initial_gene_info, d / "initial_gene_info.txt")
        st = res.states[ct]
        write_table(st.first_filter_data, d / "first_filter_data.txt", row_names=False)
        write_table(st.mapping, d / "mapping_to_initial_states.txt", row_names=False)
        _wm(st.state_abundances, d / "state_abundances.txt")
        write_table(st.state_assignment, d / "state_assignment.txt")
        write_table(st.heatmap_top_ann, d / "heatmap_top_ann.txt")
        write_table(st.gene_info, d / "gene_info.txt")
        _wm(st.heatmap_data, d / "heatmap_data.txt")
    ed = base / "Ecotypes" / "discovery"
    e = res.ecotypes
    write_table(e.mapping_all_states, ed / "mapping_all_states.txt", row_names=False)
    write_table(e.binary, ed / "binary_classification_all_states.txt")
    _wm(e.jaccard, ed / "jaccard_matrix.txt")
    write_table(pd.DataFrame({"x": [e.silhouette_initial]}), ed / "silhouette_initial.txt", row_names=False)
    write_table(e.initial_ecotypes, ed / "initial_ecotypes.txt")
    write_table(e.ecotypes, ed / "ecotypes.txt", row_names=False)
    write_table(pd.DataFrame({"x": [e.silhouette]}), ed / "silhouette.txt", row_names=False)
    write_ecotype_recovery(res.assignment, ed)
    return base


def discover_files(input_dir, annotation_path, out_dir, *, name: str, fractions: str = "All_genes",
                   scale_column=None, max_states: int = 20, restarts: int = 5, cophenetic_cutoff: float = 0.95,
                   min_states: int = 3, n_jobs: int = 1, r_exact: bool = True):
    """File-based discovery from a folder of per-cell-type profiles (CSx HiRes
    output or presorted profiles) and a sample annotation file."""
    from .rlayout import _wm  # noqa: F401

    profiles = read_profiles(input_dir)
    ann = read_annotation(annotation_path)
    res = discover_from_profiles(profiles, ann, scale_column=scale_column, max_states=max_states,
                                 restarts=restarts, cophenetic_cutoff=cophenetic_cutoff, min_states=min_states,
                                 n_jobs=n_jobs, r_exact=r_exact)
    write_discovery(res, out_dir, name, fractions)
    return res


# ----------------------------------------------------------------------------
# scRNA-seq discovery
# ----------------------------------------------------------------------------
def _nmf_rank_task_cc(V, rank, restarts):
    return _nmf_rank_task(V, rank, restarts)


def discover_scrna_frame(expr: pd.DataFrame, annotation: pd.DataFrame, *, filter_genes="cell type specific",
                         scale_column=None, max_states: int = 20, restarts: int = 5,
                         cophenetic_cutoff: float = 0.975, p_value_cutoff: float = 1.0, min_states: int = 3,
                         n_jobs: int = 1, r_exact: bool = True, verbose: bool = True):
    """``EcoTyper_discovery_scRNA.R`` on a genes x cells matrix (fread semantics)
    and a cell annotation with ``ID``, ``CellType`` and ``Sample``."""
    from . import discovery as dsc
    from . import discovery_scrna as dsr
    from .rcompat.frame import factor_levels

    cell_types = factor_levels(annotation["CellType"].astype(str).tolist())
    specific = None
    if filter_genes != "no filter":
        specific = dsr.cell_type_specific_genes(expr, annotation, cell_types, filter_genes, scale_column)
    cc = {}
    for ct in cell_types:
        if specific is not None and ct not in specific:
            continue
        try:
            genes = None if specific is None else specific[ct].filtered["Gene"].tolist()
            cc[ct] = dsr.cross_correlation(expr, annotation, ct, genes, scale_column)
        except ValueError as e:
            if verbose:
                print(e)
    tasks, keys = [], []
    for ct, c in cc.items():
        V = dsc.nmf_input(c.cor, r_exact)
        for k in range(2, max_states + 1):
            tasks.append((V, k, restarts))
            keys.append((ct, k))
    res = _run_parallel(_nmf_rank_task_cc, tasks, n_jobs)
    merged = {key: r[0] for key, r in zip(keys, res)}
    rank_tables = {ct: pd.DataFrame([res[keys.index((ct, k))][1] for k in range(2, max_states + 1)]) for ct in cc}
    ranks = dsc.select_ranks(rank_tables, max_states, cophenetic_cutoff, r_exact)
    cts = [str(c) for c in ranks["CellType"]]
    initial_cc, feats, initial, states, merged2, rank2 = {}, {}, {}, {}, {}, {}
    for ct, k in zip(cts, ranks["Chosen_Rank"]):
        k = int(k)
        initial_cc[ct] = dsc.initial_states(merged[(ct, k)], cc[ct].cor, ct, k, r_exact, gene_info=False)
        feats[ct] = dsr.extract_features_scrna(cc[ct], initial_cc[ct].initial_assignment, r_exact)
    tasks2 = [(dsc.nmf_input(feats[ct].top_scaled, r_exact), int(k), restarts) for ct, k in zip(cts, ranks["Chosen_Rank"])]
    res2 = _run_parallel(_nmf_rank_task, tasks2, n_jobs)
    for (ct, k), r in zip(zip(cts, ranks["Chosen_Rank"]), res2):
        k = int(k)
        merged2[ct], rank2[ct] = r[0], r[1]
        initial[ct] = dsc.initial_states(r[0], feats[ct].top_scaled, ct, k, r_exact)
        f = dsc.Features(ct, None, None, feats[ct].top_scaled, feats[ct].full_scaled)
        states[ct] = dsc.first_filter(initial[ct], f, annotation, r_exact, csx_filter=False)
    eco = dsr.discover_ecotypes_scrna(initial, states, cts, annotation, p_value_cutoff, min_states)
    assigned = dsr.assign_samples_scrna(initial, states, eco.ecotypes, cts, annotation)
    return DiscoveryResult(specific=specific, cross_cor=cc, merged_cc=merged, rank_tables=rank_tables, ranks=ranks,
                           initial_cc=initial_cc, features=feats, merged=merged2, rank_metrics=rank2,
                           initial=initial, states=states, ecotypes=eco, assignment=assigned,
                           params=dict(max_states=max_states, restarts=restarts, cophenetic_cutoff=cophenetic_cutoff,
                                       min_states=min_states, filter_genes=filter_genes))


def discover_scrna(adata, *, layer=None, **kwargs):
    """scRNA-seq discovery on AnnData (cells x genes, raw-scale expression;
    ``obs`` with ``CellType`` and ``Sample``). Results are stored in
    ``adata.obs["ecotyper_state_<cell type>"]`` (per cell) and returned."""
    expr = from_anndata(adata, layer)
    ann = annotation_from_anndata(adata)
    res = discover_scrna_frame(expr, ann, **kwargs)
    for ct, st in res.states.items():
        s = pd.Series(np.nan, index=adata.obs_names, dtype=object)
        a = st.state_assignment
        s.loc[a["ID"].astype(str).to_numpy()] = a["State"].to_numpy()
        adata.obs[f"ecotyper_state_{ct}"] = pd.Categorical(s)
    adata.uns["ecotyper_discovery"] = {"ranks": res.ranks, "ecotypes": res.ecotypes.ecotypes}
    return res


def write_discovery_scrna(res, out_dir, name: str, fractions: str = "Cell_type_specific_genes",
                          write_large: bool = False) -> Path:
    """Write the R layout of a scRNA discovery run. The very large
    ``expression_full_matrix_*`` files are only written with ``write_large``."""
    from .rcompat.io import format_real, write_table

    base = Path(out_dir) / "EcoTyper" / name / fractions
    p = res.params
    if res.specific is not None:
        d = base / "Analysis" / "Cell_type_specific_genes"
        for ct, sg in res.specific.items():
            write_table(sg.raw, d / f"{ct}_cell_type_specific_genes_raw.txt", row_names=False)
            write_table(sg.filtered, d / f"{ct}_cell_type_specific_genes.txt", row_names=False)
    ccd = base / "Cell_States" / "discovery_cross_cor"
    for ct, c in res.cross_cor.items():
        d = ccd / ct
        d.mkdir(parents=True, exist_ok=True)
        if write_large:
            _wm(c.log_data, d / "expression_full_matrix_log2.txt")
        _wm(c.scaled_filt, d / "expression_top_genes_scaled_filt.txt")
        _wm(c.cor, d / "expression_top_genes_scaled.txt")
        for k in range(2, p["max_states"] + 1):
            row = res.rank_tables[ct].iloc[[k - 2]].reset_index(drop=True)
            row["n_clusters"] = row["n_clusters"].astype(int)
            write_table(row, d / str(k) / "rank_data.txt", row_names=False)
    rs = base / "Analysis" / "rank_selection"
    write_table(res.ranks, rs / "rank_data.txt", row_names=False)
    write_table(res.ranks, rs / f"rank_data_cutoff_{format_real(p['cophenetic_cutoff'])}.txt", row_names=False)
    for ct, ini in res.initial_cc.items():
        d = ccd / ct / str(ini.rank)
        _wm(ini.W, d / "W.txt")
        _wm(ini.H, d / "H.txt")
        _wm(ini.initial_abundances, d / "initial_state_abundances.txt")
        write_table(ini.initial_assignment, d / "initial_state_assignment.txt", row_names=False)
    cs = base / "Cell_States" / "discovery"
    for ct, f in res.features.items():
        d = cs / ct
        write_table(f.full_fcs, d / "full_FCs.txt")
        write_table(f.top_fcs, d / "top_FCs.txt")
        if write_large:
            _wm(f.log_full, d / "expression_full_matrix_log2.txt")
        _wm(f.log_top, d / "expression_top_genes_log2.txt")
        _wm(f.top_scaled, d / "expression_top_genes_scaled.txt")
        _wm(f.full_scaled, d / "expression_full_matrix_scaled.txt")
        ini = res.initial[ct]
        row = pd.DataFrame([res.rank_metrics[ct]])
        row["n_clusters"] = row["n_clusters"].astype(int)
        e = cs / ct / str(ini.rank)
        write_table(row, e / "rank_data.txt", row_names=False)
        _wm(ini.W, e / "W.txt")
        _wm(ini.H, e / "H.txt")
        _wm(ini.initial_abundances, e / "initial_state_abundances.txt")
        write_table(ini.initial_assignment, e / "initial_state_assignment.txt", row_names=False)
        write_table(ini.initial_gene_info, e / "initial_gene_info.txt")
        st = res.states[ct]
        write_table(st.first_filter_data, e / "first_filter_data.txt", row_names=False)
        write_table(st.mapping, e / "mapping_to_initial_states.txt", row_names=False)
        _wm(st.state_abundances, e / "state_abundances.txt")
        write_table(st.state_assignment, e / "state_assignment.txt")
        write_table(st.heatmap_top_ann, e / "heatmap_top_ann.txt")
        write_table(st.gene_info, e / "gene_info.txt")
        _wm(st.heatmap_data, e / "heatmap_data.txt")
    ed = base / "Ecotypes" / "discovery"
    eco = res.ecotypes
    write_table(eco.mapping_all_states, ed / "mapping_all_states.txt", row_names=False)
    write_table(eco.binary, ed / "binary_classification_all_states.txt")
    _wm(eco.jaccard, ed / "jaccard_matrix.txt")
    write_table(pd.DataFrame({"x": [eco.silhouette_initial]}), ed / "silhouette_initial.txt", row_names=False)
    write_table(eco.initial_ecotypes, ed / "initial_ecotypes.txt")
    write_table(eco.ecotypes, ed / "ecotypes.txt", row_names=False)
    write_table(pd.DataFrame({"x": [eco.silhouette]}), ed / "silhouette.txt", row_names=False)
    write_ecotype_recovery(res.assignment, ed)
    return base
