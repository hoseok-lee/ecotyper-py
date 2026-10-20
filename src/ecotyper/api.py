"""scverse-style entry points: :func:`recovery`, :func:`discovery`, :func:`infer_data_type`.

These take :class:`anndata.AnnData` objects as produced by scanpy/squidpy
(``sc.read_10x_mtx``, ``sc.read_visium``, ``sc.read_h5ad``…), with dense or
sparse ``X`` or a ``layer``, and store results in ``adata.obs``/``obsm``/``uns``
under ``key_added`` (default ``"ecotyper"``). They wrap the exact engine in
:mod:`ecotyper.workflows`, so results equal the file-based workflows (and R).

Data types
    ``"bulk"``     samples x genes (bulk RNA-seq or microarray);
    ``"scRNA"``    cells x genes, raw-scale (counts/TPM), ``obs`` with a cell type column;
    ``"visium"``   spots x genes (10x Visium), with CIBERSORTx cell type fractions;
    ``"profiles"`` (discovery only) per-cell-type expression profiles: a dict
    ``{cell type: AnnData (samples x genes)}`` (CIBERSORTx HiRes output or
    presorted/purified populations).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .model import EcoTyperModel

DataType = Literal["bulk", "scRNA", "visium", "profiles"]
_DATA_TYPES = ("bulk", "scRNA", "visium", "profiles")


# ----------------------------------------------------------------------------
# data type inference
# ----------------------------------------------------------------------------
def _is_spatial(adata) -> bool:
    return "spatial" in adata.obsm or "spatial" in adata.uns


def infer_data_type(adata, *, cell_type_key: str = "CellType") -> DataType:
    """Guess the data type of ``adata``.

    * a ``dict`` of per-cell-type objects → ``"profiles"``;
    * ``obsm["spatial"]`` or ``uns["spatial"]`` (squidpy/scanpy Visium) → ``"visium"``;
    * ``obs[cell_type_key]`` present → ``"scRNA"``;
    * otherwise → ``"bulk"``.

    Raises ``ValueError`` when the signals conflict (spatial coordinates *and* a
    per-observation cell type column, as in imaging-based single-cell spatial
    data) or when the data look single-cell (sparse, many observations) but
    there is no cell type column. Pass ``data_type=`` explicitly in those cases.
    """
    if isinstance(adata, dict):
        return "profiles"
    spatial = _is_spatial(adata)
    has_ct = cell_type_key in adata.obs.columns
    if spatial and has_ct:
        raise ValueError(
            f"Cannot tell the data type: adata has spatial coordinates and a per-observation cell type column "
            f"obs[{cell_type_key!r}]. Pass data_type='visium' (spots with CIBERSORTx fractions) or "
            f"data_type='scRNA' (annotated single cells).")
    if spatial:
        return "visium"
    if has_ct:
        return "scRNA"
    import scipy.sparse as sp

    X = adata.X
    if (sp.issparse(X) or type(X).__name__.endswith("_array")) and adata.n_obs > 2000:
        raise ValueError(
            f"adata looks like single-cell data (sparse X, {adata.n_obs} observations) but obs has no "
            f"{cell_type_key!r} column. Pass cell_type_key=<obs column with cell type labels>, or "
            f"data_type='bulk' if these are bulk samples.")
    return "bulk"


def _resolve_type(adata, data_type, cell_type_key) -> str:
    if data_type is None:
        return infer_data_type(adata, cell_type_key=cell_type_key)
    aliases = {"scrna": "scRNA", "sc": "scRNA", "single-cell": "scRNA", "single_cell": "scRNA",
               "spatial": "visium", "presorted": "profiles", "purified": "profiles"}
    dt = aliases.get(str(data_type).lower(), str(data_type))
    if dt == "scrna":
        dt = "scRNA"
    if dt not in _DATA_TYPES:
        raise ValueError(f"data_type must be one of {_DATA_TYPES}, got {data_type!r}")
    return dt


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _model(model) -> EcoTyperModel:
    if isinstance(model, EcoTyperModel):
        return model
    return EcoTyperModel.load(model)


def _matrix(adata, layer):
    X = adata.layers[layer] if layer is not None else adata.X
    if X is None:
        raise ValueError("adata.X is empty; pass layer=<name of the layer with expression values>")
    return X


def _expr_frame(adata, layer, genes=None, obs_names=None) -> pd.DataFrame:
    """genes x observations DataFrame (dense), optionally only ``genes`` (first
    occurrence of each name, in var order)."""
    import scipy.sparse as sp

    X = _matrix(adata, layer)
    var = [str(g) for g in adata.var_names]
    if genes is not None:
        want = set(genes)
        seen = set()
        keep = []
        for i, g in enumerate(var):
            if g in want and g not in seen:
                seen.add(g)
                keep.append(i)
        X = X[:, keep]
        var = [var[i] for i in keep]
    if sp.issparse(X):
        X = X.toarray()
    elif hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    cols = list(adata.obs_names) if obs_names is None else list(obs_names)
    return pd.DataFrame(np.ascontiguousarray(X.T), index=pd.Index(var, dtype=object), columns=cols)


def _min_value(adata, layer) -> float:
    import scipy.sparse as sp

    X = _matrix(adata, layer)
    if sp.issparse(X):
        d = X.data
        m = float(d.min()) if d.size else 0.0
        return min(m, 0.0) if X.nnz < X.shape[0] * X.shape[1] else m
    return float(np.nanmin(np.asarray(X)))


def _obs_annotation(adata, renames: dict) -> pd.DataFrame:
    """``adata.obs`` as an R-style annotation table (``ID`` first; ``renames``
    maps user obs columns to the column names EcoTyper uses)."""
    obs = adata.obs.copy()
    for src, dst in renames.items():
        if src is None or src == dst:
            continue
        if src not in obs.columns:
            raise KeyError(f"obs has no column {src!r}")
        if dst in obs.columns:
            obs = obs.drop(columns=[dst])
        obs = obs.rename(columns={src: dst})
    obs = obs.drop(columns=[c for c in ("ID",) if c in obs.columns])
    obs.insert(0, "ID", [str(i) for i in adata.obs_names])
    out = obs.reset_index(drop=True)
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object)
    return out


def _state_columns(adata, key, states: dict, id_to_obs: dict | None = None) -> None:
    for ct, r in states.items():
        ab = r.abundances.T
        if id_to_obs is not None:
            ab.index = [id_to_obs.get(str(i), str(i)) for i in ab.index]
        ab = ab.reindex(adata.obs_names)
        ab.columns = [str(c) for c in ab.columns]
        adata.obsm[f"{key}_abundance_{ct}"] = ab
        st = pd.Series(np.nan, index=adata.obs_names, dtype=object)
        ids = [str(i) for i in r.assignment["ID"]]
        if id_to_obs is not None:
            ids = [id_to_obs.get(i, i) for i in ids]
        st.loc[ids] = r.assignment["State"].to_numpy()
        adata.obs[f"{key}_state_{ct}"] = pd.Categorical(st, categories=list(r.abundances.index))


def _cell_ecotypes(adata, key, model: EcoTyperModel, cell_types) -> None:
    """Per-cell ecotype: the ecotype containing the cell's state (NaN otherwise)."""
    if model.ecotypes is None:
        return
    eco = model.ecotypes
    lut = {(str(c), str(s)): str(e) for c, s, e in zip(eco["CellType"], eco["State"], eco["Ecotype"])}
    out = pd.Series(np.nan, index=adata.obs_names, dtype=object)
    for ct in cell_types:
        col = f"{key}_state_{ct}"
        if col not in adata.obs:
            continue
        st = adata.obs[col]
        ok = st.notna()
        out[ok] = [lut.get((ct, str(s)), np.nan) for s in st[ok]]
    adata.obs[f"{key}_ecotype"] = pd.Categorical(out, categories=model.ecotype_names)


def _uns(adata, key, **entries) -> None:
    adata.uns[key] = {k: v for k, v in entries.items() if v is not None}


# ----------------------------------------------------------------------------
# recovery
# ----------------------------------------------------------------------------
def recovery(adata, model: EcoTyperModel | str = "Carcinoma", *, data_type: DataType | None = None,
             layer: str | None = None, cell_type_key: str = "CellType", sample_key: str | None = "Sample",
             fractions: pd.DataFrame | str | None = None, malignant_cell: str | None = "Epithelial.cells",
             library_id: str | None = None, subsample: int = -1, key_added: str = "ecotyper",
             copy: bool = False, seed: int = 42, n_jobs: int = 1, r_exact: bool = True, exact: bool = True):
    """Recover cell states and ecotypes defined by ``model`` in new data.

    Parameters
    ----------
    adata
        Observations (samples, cells or spots) x genes. ``X`` (or ``layer``)
        holds expression on the raw scale (counts, TPM, or normalised
        microarray values); dense and sparse matrices are accepted. Bulk data
        with values > 50 are log2(x+1)-transformed as in R.
    model
        An :class:`EcoTyperModel` (e.g. from :func:`discovery`), a built-in
        model name (``"Carcinoma"``, ``"Lymphoma"``) or a path to a saved model.
    data_type
        ``"bulk"``, ``"scRNA"`` or ``"visium"``; inferred with
        :func:`infer_data_type` when omitted.
    cell_type_key
        scRNA: ``obs`` column with cell type labels. They must match the
        model's cell type names (``model.cell_type_order``); cells of other
        types are ignored.
    sample_key
        scRNA: ``obs`` column with the sample of each cell. Needed for
        ecotype abundances per sample.
    fractions
        Visium: CIBERSORTx cell type fractions per spot, a DataFrame
        (index: ``obs_names``; columns: cell types, extra CIBERSORTx columns
        such as ``P.value`` are dropped) or the name of an ``obsm`` entry.
    malignant_cell
        Visium: the cell type treated as malignant/background fraction (R's
        ``Malignant cell type``); ``None`` for none.
    library_id
        Visium: sample name used in the output (default: the single key of
        ``uns["spatial"]``, else ``"Visium"``).
    subsample
        scRNA: maximum number of cells per cell type (R's ``-s``; values below
        50 keep all cells).
    key_added
        Prefix of the result keys.
    copy
        Return a modified copy instead of writing into ``adata``.
    seed, n_jobs, r_exact
        Random seed of NMF prediction (R's ``set.seed``), number of worker
        processes, and R-exact arithmetic (keep ``True`` for results identical to R).
    exact
        ``False`` selects the fast mode: NMF with BLAS matrix products and no
        emulation of R's text round trips. Results match R's to ~1e-12 relative
        and state/ecotype assignments were identical on the tutorial data, but
        bit-identity with R is not guaranteed (see the docs, "Fast mode").

    Returns
    -------
    ``None`` (or the copy with ``copy=True``). Results:

    ``obs[f"{key}_state_{cell type}"]``
        cell state per observation (bulk/Visium: dominant state; scRNA: the
        state of each cell of that type, NaN for other cells);
    ``obsm[f"{key}_abundance_{cell type}"]``
        cell state abundances (columns ``S01``…);
    ``obs[f"{key}_ecotype"]``
        bulk: assigned ecotype (``Unassigned`` if not significant), with
        ``obs[f"{key}_max_ecotype"]`` and ``obs[f"{key}_ecotype_q"]``;
        scRNA: ecotype of each cell's state;
    ``obsm[f"{key}_ecotype_abundance"]``
        bulk and Visium: ecotype abundances per observation;
    ``obsm[f"{key}_state_abundance"]``
        Visium: fraction-weighted state abundances per spot (columns
        ``<cell type>_<state>``);
    ``uns[key]``
        parameters, NMF iterations per cell type, and for scRNA the ecotype
        abundances (``ecotype_abundance``, samples x ecotypes) and
        assignment (``ecotype_assignment``) per sample.
    """
    from . import workflows as wf

    if copy:
        adata = adata.copy()
    dt = _resolve_type(adata, data_type, cell_type_key)
    model = _model(model)
    key = key_added
    params = dict(model=model.name, fractions=model.fractions, data_type=dt, seed=seed, layer=layer, exact=exact)

    if dt == "bulk":
        expr = _expr_frame(adata, layer)
        ann = _obs_annotation(adata, {})
        states, eco = wf.recover_bulk_frame(expr, model, ann, seed=seed, n_jobs=n_jobs, r_exact=r_exact,
                                            exact=exact)
        _state_columns(adata, key, states)
        if eco is not None:
            ia = eco.initial_assignment.set_index("ID")
            idx = adata.obs_names
            adata.obs[f"{key}_ecotype"] = pd.Categorical(
                ia["Ecotype"].reindex(idx), categories=list(eco.ecotype_abundance.index) + ["Unassigned"])
            adata.obs[f"{key}_max_ecotype"] = pd.Categorical(
                ia["MaxEcotype"].reindex(idx), categories=list(eco.ecotype_abundance.index))
            adata.obs[f"{key}_ecotype_q"] = ia["AssignmentQ"].reindex(idx).to_numpy(dtype=float)
            ea = eco.ecotype_abundance.T.reindex(idx)
            ea.columns = [str(c) for c in ea.columns]
            adata.obsm[f"{key}_ecotype_abundance"] = ea
        _uns(adata, key, params=params, niter={ct: int(r.niter) for ct, r in states.items()})

    elif dt == "scRNA":
        if cell_type_key not in adata.obs.columns:
            raise KeyError(f"obs has no cell type column {cell_type_key!r}; pass cell_type_key=")
        if _min_value(adata, layer) < 0:
            raise ValueError("Not all values in the expression matrix are positive; scRNA recovery expects "
                             "raw-scale (non-log, non-scaled) expression. Use layer= to select counts.")
        renames = {cell_type_key: "CellType"}
        if sample_key is not None and sample_key in adata.obs.columns:
            renames[sample_key] = "Sample"
        ann = _obs_annotation(adata, renames)
        ann["CellType"] = ann["CellType"].astype(str)
        present = set(ann["CellType"])
        missing = [ct for ct in model.cell_type_order if ct not in present]
        if len(missing) == len(model.cell_type_order):
            raise ValueError(f"None of the model's cell types {model.cell_type_order} occur in "
                             f"obs[{cell_type_key!r}] (values: {sorted(present)[:10]}…). Rename the labels "
                             f"to the model's cell type names.")
        # recovery only uses the model genes: densify just those (exact: same rows R selects)
        genes = set()
        for ctm in model.cell_types.values():
            genes.update(str(g) for g in ctm.initial_gene_info["Gene"])
        expr = _expr_frame(adata, layer, genes=genes)
        states, eco = wf.recover_scrna_frame(expr, ann, model, seed=seed, subsample=subsample, n_jobs=n_jobs,
                                             exact=exact)
        _state_columns(adata, key, states)
        _cell_ecotypes(adata, key, model, list(states))
        ea = asg = None
        if eco is not None:
            ea = eco.ecotype_abundance.T.copy()
            ea.columns = [str(c) for c in ea.columns]
            ea.index = [str(i) for i in ea.index]
            asg = eco.initial_assignment.set_index("ID")
            asg.index = [str(i) for i in asg.index]
            asg = asg.astype({c: str for c in asg.columns if asg[c].dtype == object})
        _uns(adata, key, params={**params, "cell_type_key": cell_type_key, "sample_key": sample_key},
             niter={ct: int(r.niter) for ct, r in states.items()}, ecotype_abundance=ea, ecotype_assignment=asg,
             missing_cell_types=missing or None)

    elif dt == "visium":
        from .rcompat.frame import make_names

        if fractions is None:
            raise ValueError("Visium recovery needs CIBERSORTx cell type fractions per spot: pass fractions=<DataFrame "
                             "indexed by obs_names> or the name of an obsm entry")
        frac = adata.obsm[fractions] if isinstance(fractions, str) else fractions
        frac = pd.DataFrame(frac).copy()
        if "Mixture" in frac.columns:
            frac = frac.set_index("Mixture")
        cols = make_names([str(c) for c in frac.columns])
        if "P.value" in cols:
            frac = frac.iloc[:, : cols.index("P.value")]
        ids = make_names([str(i) for i in adata.obs_names])
        id_to_obs = dict(zip(ids, (str(i) for i in adata.obs_names)))
        frac.index = make_names([str(i) for i in frac.index])
        frac = frac.reset_index(names="Mixture")
        for k in ("array_row", "array_col"):
            if k not in adata.obs.columns:
                raise KeyError(f"Visium recovery needs obs[{k!r}] (array coordinates, as set by sc.read_visium)")
        spots = pd.DataFrame({"ID": ids, "X": adata.obs["array_row"].to_numpy(dtype=float),
                              "Y": adata.obs["array_col"].to_numpy(dtype=float)})
        if library_id is None:
            libs = list(adata.uns.get("spatial", {}).keys())
            library_id = libs[0] if len(libs) == 1 else "Visium"
        expr = _expr_frame(adata, layer, obs_names=ids)
        states, long, wide, eco = wf.recover_visium_frame(expr, spots, model, frac, sample_name=library_id,
                                                          malignant_cell=malignant_cell, seed=seed,
                                                          n_jobs=n_jobs, r_exact=r_exact, exact=exact)
        _state_columns(adata, key, states, id_to_obs)
        fixed = ["ID", "X", "Y", "Sample", "Malignant"]
        for name, tab in ((f"{key}_state_abundance", wide), (f"{key}_ecotype_abundance", eco)):
            t = tab.copy()
            t.index = [id_to_obs.get(str(i), str(i)) for i in t["ID"]]
            t = t[[c for c in t.columns if c not in fixed]].reindex(adata.obs_names)
            t.columns = [str(c) for c in t.columns]
            adata.obsm[name] = t.astype(float)
        mal = wide.set_index("ID")["Malignant"]
        mal.index = [id_to_obs.get(str(i), str(i)) for i in mal.index]
        adata.obs[f"{key}_malignant_fraction"] = mal.reindex(adata.obs_names).to_numpy(dtype=float)
        _uns(adata, key, params={**params, "library_id": library_id, "malignant_cell": malignant_cell},
             niter={ct: int(r.niter) for ct, r in states.items()})
    else:
        raise ValueError(f"recovery does not support data_type={dt!r}")
    return adata if copy else None


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------
def discovery(data, *, data_type: DataType | None = None, layer: str | None = None,
              cell_type_key: str = "CellType", sample_key: str = "Sample", annotation: pd.DataFrame | None = None,
              name: str = "custom", filter_genes: str | int = "cell type specific", scale_column: str | None = None,
              max_states: int = 20, restarts: int = 5, cophenetic_cutoff: float | None = None,
              p_value_cutoff: float = 1.0, min_states: int = 3, key_added: str = "ecotyper", n_jobs: int = 1,
              r_exact: bool = True, verbose: bool = True, exact: bool = True,
              nmf_method: str = "brunet") -> EcoTyperModel:
    """Discover cell states and ecotypes; returns an in-memory :class:`EcoTyperModel`.

    Parameters
    ----------
    data
        * scRNA-seq: AnnData (cells x genes, raw-scale expression) with cell
          type labels in ``obs[cell_type_key]`` and samples in
          ``obs[sample_key]``;
        * per-cell-type profiles: ``{cell type: AnnData}`` (samples x genes,
          the same samples in every object), e.g. CIBERSORTx HiRes purified
          profiles or presorted populations. Sample annotation from
          ``annotation`` (column ``ID`` plus covariates) or the ``obs`` of the
          first object.
    data_type
        ``"scRNA"`` or ``"profiles"``; inferred when omitted. Bulk discovery
        from mixtures needs CIBERSORTx (not part of this package): run it and
        pass the HiRes profiles as ``"profiles"``.
    filter_genes
        scRNA gene filter (R's ``Filter genes``): ``"cell type specific"``,
        ``"no filter"``, or the number of top variable genes.
    scale_column
        Annotation column to scale within (R's ``Scale column``), e.g. a batch.
    max_states, restarts, cophenetic_cutoff, p_value_cutoff, min_states
        NMF and ecotype parameters as in the R configuration files.
        ``cophenetic_cutoff`` defaults to 0.975 (scRNA) and 0.95 (profiles).
    key_added
        scRNA: prefix of the per-cell results written to ``data.obs``.
    exact
        ``False`` selects the fast mode (BLAS NMF updates, no R text round
        trips); see :func:`recovery`.
    nmf_method
        ``"brunet"`` (default, EcoTyper's KL multiplicative updates) or
        ``"cd"``: Frobenius-loss coordinate descent from scikit-learn, an
        opt-in extension that is not equivalent to EcoTyper (states can
        differ; see the docs, "Alternative NMF solver").

    Notes
    -----
    The model keeps the full discovery result in ``model.discovery_result``.
    ``model.save(path)`` writes it in the R layout; for the complete R output
    tree use :func:`ecotyper.workflows.write_discovery_scrna` /
    :func:`ecotyper.workflows.write_discovery` on ``model.discovery_result``.
    """
    from . import workflows as wf

    dt = _resolve_type(data, data_type, cell_type_key)
    if dt == "bulk":
        raise ValueError(
            "Bulk discovery starts from CIBERSORTx: deconvolve the mixtures with CIBERSORTx HiRes, then pass the "
            "per-cell-type profiles as data={cell type: AnnData} (data_type='profiles').")
    if dt == "visium":
        raise ValueError("EcoTyper does not discover states from Visium data; discover from scRNA-seq or bulk "
                         "profiles and use ecotyper.recovery on the Visium data.")
    if dt == "scRNA":
        adata = data
        for k in (cell_type_key, sample_key):
            if k not in adata.obs.columns:
                raise KeyError(f"obs has no column {k!r} (scRNA discovery needs cell types and samples)")
        if _min_value(adata, layer) < 0:
            raise ValueError("scRNA discovery expects raw-scale (non-log, non-scaled) expression; use layer=")
        ann = _obs_annotation(adata, {cell_type_key: "CellType", sample_key: "Sample"})
        ann["CellType"] = ann["CellType"].astype(str)
        expr = _expr_frame(adata, layer)
        res = wf.discover_scrna_frame(expr, ann, filter_genes=filter_genes, scale_column=scale_column,
                                      max_states=max_states, restarts=restarts,
                                      cophenetic_cutoff=0.975 if cophenetic_cutoff is None else cophenetic_cutoff,
                                      p_value_cutoff=p_value_cutoff, min_states=min_states, n_jobs=n_jobs,
                                      r_exact=r_exact, verbose=verbose, exact=exact, nmf_method=nmf_method)
        model = EcoTyperModel.from_discovery(res, name)
        key = key_added
        for ct, st in res.states.items():
            s = pd.Series(np.nan, index=adata.obs_names, dtype=object)
            a = st.state_assignment
            s.loc[a["ID"].astype(str).to_numpy()] = a["State"].to_numpy()
            adata.obs[f"{key}_state_{ct}"] = pd.Categorical(s, categories=list(st.mapping["State"]))
        _cell_ecotypes(adata, key, model, list(res.states))
        _uns(adata, key, params=dict(data_type="scRNA", name=name, filter_genes=str(filter_genes),
                                     max_states=max_states, restarts=restarts, layer=layer,
                                     cell_type_key=cell_type_key, sample_key=sample_key, exact=exact,
                                     nmf_method=nmf_method),
             n_states=model.n_states)
        return model
    # profiles
    if not isinstance(data, dict) or not data:
        raise TypeError("profiles must be a non-empty dict {cell type: AnnData (samples x genes)}")
    profiles = {}
    first = None
    for ct, a in data.items():
        if isinstance(a, pd.DataFrame):
            profiles[str(ct)] = a  # genes x samples, as read from CIBERSORTx output
            continue
        first = a if first is None else first
        profiles[str(ct)] = _expr_frame(a, layer)
    if annotation is None and first is not None and first.obs.shape[1] > 0:
        annotation = _obs_annotation(first, {})
    res = wf.discover_from_profiles(profiles, annotation, scale_column=scale_column, max_states=max_states,
                                    restarts=restarts,
                                    cophenetic_cutoff=0.95 if cophenetic_cutoff is None else cophenetic_cutoff,
                                    min_states=min_states, n_jobs=n_jobs, r_exact=r_exact, verbose=verbose,
                                    exact=exact, nmf_method=nmf_method)
    return EcoTyperModel.from_discovery(res, name)
