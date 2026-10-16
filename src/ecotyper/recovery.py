"""Cell state and ecotype recovery (``pipeline/state_recovery_*.R``,
``ecotypes_recovery*.R``, ``spatial_states.R``, ``spatial_ecotypes.R``).

All functions take and return pandas objects laid out like the R objects
(genes x samples matrices, data frames with an ``ID`` column) so every
intermediate file of the R pipeline can be reproduced. ``r_exact=True``
reproduces the places where the R pipeline writes a table with 15
significant digits and the next script reads it back, which is necessary for
bit-identical results downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import CellTypeModel, ecotype_levels
from .nmf import nmf_predict
from .rcompat.frame import factor_levels, make_names, r_match, r_merge, r_order
from .rcompat.parse import text_roundtrip
from .rcompat.rng import RRandom
from .rcompat.stats import (TTestError, apply_mean_rows, col_apply_mean, col_normalize, log2p1,
                            p_adjust_bh, quantile7, scale_rows, which_max, welch_t_test_p)


def _rt(x, r_exact: bool):
    return text_roundtrip(x, "r") if r_exact else np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------
# input preparation
# ----------------------------------------------------------------------------
def prepare_bulk_expression(expr: pd.DataFrame) -> pd.DataFrame:
    """``data[is.na(data)] = 0``; log2(x + 1) if all values >= 0 and max > 50."""
    X = expr.to_numpy(dtype=np.float64, copy=True)
    X[np.isnan(X)] = 0.0
    if (X >= 0).all() and X.max() > 50:
        X = log2p1(X)
    return pd.DataFrame(X, index=expr.index, columns=expr.columns)


def align_rows(expr: pd.DataFrame, genes) -> pd.DataFrame:
    """``data[match(genes, rownames(data)), ]; rownames = genes; NA -> 0``."""
    genes = list(genes)
    idx = r_match(genes, list(expr.index))
    X = np.zeros((len(genes), expr.shape[1]))
    ok = idx >= 0
    X[ok] = expr.to_numpy(dtype=np.float64)[idx[ok]]
    X[np.isnan(X)] = 0.0
    return pd.DataFrame(X, index=pd.Index(genes, dtype=object), columns=expr.columns)


def read_clinical(sample_list, annotation: pd.DataFrame | None) -> pd.DataFrame:
    """``read_clinical`` (lib/misc.R): annotation rows matched to the samples."""
    sample_list = list(sample_list)
    if annotation is None:
        return pd.DataFrame({"ID": sample_list}, index=pd.Index(make_names(sample_list, True), dtype=object))
    idx = r_match(sample_list, list(annotation["ID"]))
    cols = {}
    for c in annotation.columns:
        v = annotation[c].to_numpy(dtype=object)
        cols[c] = np.array([v[i] if i >= 0 else None for i in idx], dtype=object)
    cl = pd.DataFrame(cols)
    cl["ID"] = np.array(sample_list, dtype=object)
    cl.index = pd.Index(make_names(sample_list, True), dtype=object)
    return cl


# ----------------------------------------------------------------------------
# cell state recovery
# ----------------------------------------------------------------------------
@dataclass
class StateRecovery:
    cell_type: str
    expression: pd.DataFrame  # genes (initial_gene_info) x samples, unscaled
    scaled: pd.DataFrame
    H: pd.DataFrame  # initial states x samples (raw NMF coefficients)
    initial_abundances: pd.DataFrame
    initial_assignment: pd.DataFrame  # ID, State
    abundances: pd.DataFrame  # states (S01..) x samples
    assignment: pd.DataFrame  # ID, State, InitialState
    niter: int
    annotation: pd.DataFrame | None = None
    heatmap_data: pd.DataFrame | None = None
    heatmap_top_ann: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


def _sort_assignment(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[r_order(df["State"].tolist())]


def recover_states(expr: pd.DataFrame, ct: CellTypeModel, rng: RRandom,
                   annotation: pd.DataFrame | None = None, *, kind: str = "bulk") -> StateRecovery:
    """Recover the cell states of one cell type (``state_recovery_{bulk,visium,scRNA}.R``).

    ``expr`` is the genes x samples matrix after the dataset-level preparation
    (NA -> 0 and optional log2 for bulk/Visium via
    :func:`prepare_bulk_expression`; per-cell-type log2 for scRNA). ``rng`` is
    consumed exactly as R consumes its RNG (one ``rnmf`` draw).
    ``annotation`` is the result of :func:`read_clinical` (bulk/Visium) or the
    cell annotation (scRNA).
    """
    igi_genes = [str(g) for g in ct.initial_gene_info["Gene"]]
    data = align_rows(expr, igi_genes)
    unscaled = data
    scaled = scale_rows(data.to_numpy())
    scaled[np.isnan(scaled)] = 0.0
    scaled_df = pd.DataFrame(scaled, index=data.index, columns=data.columns)

    # NMFpredict: training genes are the "__pos" rows of W
    wrows = [str(r) for r in ct.W.index]
    training = [r.replace("__pos", "") for r in wrows if "__pos" in r]
    new_data = align_rows(scaled_df, training).to_numpy()
    H, niter, _ = nmf_predict(ct.W.to_numpy(), new_data, rng)
    states0 = [str(c) for c in ct.W.columns]
    samples = list(data.columns)
    H_df = pd.DataFrame(H, index=states0, columns=samples)
    ab0 = col_normalize(H)
    ab0_df = pd.DataFrame(ab0, index=states0, columns=samples)

    best = [states0[which_max(ab0[:, j])] for j in range(ab0.shape[1])]
    ia = pd.DataFrame({"ID": samples, "State": best})
    ia = _sort_assignment(ia)

    mp_init = [str(s) for s in ct.mapping["InitialState"]]
    mp_state = [str(s) for s in ct.mapping["State"]]
    ridx = r_match(mp_init, states0)
    ab = np.full((len(mp_init), ab0.shape[1]), np.nan)
    ok = ridx >= 0
    ab[ok] = ab0[ridx[ok]]
    ab_df = pd.DataFrame(ab, index=mp_state, columns=samples)

    a = ia.copy()
    a["InitialState"] = a["State"]
    m = r_match(a["InitialState"].tolist(), mp_init)
    a["State"] = [mp_state[k] if k >= 0 else None for k in m]
    a = a[pd.notna(a["State"]).to_numpy()]
    a = _sort_assignment(a)

    res = StateRecovery(cell_type=ct.cell_type, expression=unscaled, scaled=scaled_df, H=H_df,
                        initial_abundances=ab0_df, initial_assignment=ia, abundances=ab_df,
                        assignment=a, niter=niter, annotation=annotation)

    # heatmap tables (heatmap_data.txt / heatmap_top_ann.txt)
    if annotation is not None:
        top = r_merge(a.reset_index(drop=True), annotation.reset_index(drop=True), "ID", all_x=True)
    else:
        top = r_merge(a.reset_index(drop=True), pd.DataFrame({"ID": samples}), "ID", all_x=True)
    top = top.iloc[r_order(top["State"].tolist())]
    top.index = pd.Index(top["ID"].tolist(), dtype=object)
    res.heatmap_top_ann = top
    if ct.heatmap_genes is not None:
        # R: match(rownames(discovery_data), rownames(.)) without NA -> 0
        gidx = r_match(ct.heatmap_genes, list(unscaled.index))
        Xh = np.full((len(gidx), len(top)), np.nan)
        cidx = r_match(top["ID"].tolist(), samples)
        U = unscaled.to_numpy()
        for i, g in enumerate(gidx):
            if g >= 0:
                Xh[i] = U[g, cidx]
        res.heatmap_data = pd.DataFrame(Xh, index=pd.Index(ct.heatmap_genes, dtype=object),
                                        columns=top["ID"].tolist())
    return res


# ----------------------------------------------------------------------------
# recovery significance (-z TRUE)
# ----------------------------------------------------------------------------
def average_fc_marker_genes(data: pd.DataFrame, gene_info: pd.DataFrame, assignment: pd.DataFrame) -> pd.Series:
    """``averge_FC_marker_genes`` (lib/ecotyper.R)."""
    genes = [str(g) for g in gene_info["Gene"]]
    gi = r_match(genes, list(data.index))
    si = r_match(assignment["ID"].tolist(), list(data.columns))
    D = data.to_numpy()
    X = np.full((len(gi), len(si)), np.nan)
    for a, g in enumerate(gi):
        if g >= 0:
            X[a] = np.where(si >= 0, D[g, np.maximum(si, 0)], np.nan)
    st = np.array([str(s) for s in assignment["State"]], dtype=object)
    levels = factor_levels(gene_info["State"].tolist())
    fc = {}
    for x in levels:
        inx = st == x
        if inx.sum() == 0:
            fc[x] = np.full(len(gi), np.nan)
        else:
            fc[x] = apply_mean_rows(X[:, inx], na_rm=True) - apply_mean_rows(X[:, ~inx], na_rm=True)
    gstate = np.array([str(s) for s in gene_info["State"]], dtype=object)
    out = {}
    from .rcompat.stats import r_mean
    for x in levels:
        sel = gstate == x
        if sel.sum() < 2:
            out[x] = np.nan
        else:
            out[x] = r_mean(fc[x][sel], na_rm=True)
    return pd.Series(out)


# ----------------------------------------------------------------------------
# ecotype recovery
# ----------------------------------------------------------------------------
@dataclass
class EcotypeRecovery:
    combined_state_abundances: pd.DataFrame
    ecotype_abundance: pd.DataFrame
    assignment_p_vals: pd.DataFrame
    initial_assignment: pd.DataFrame  # initial_ecotype_assignment.txt
    assignment: pd.DataFrame  # ecotype_assignment.txt
    ecotypes: pd.DataFrame


def _sorted_ecotypes(ecotypes: pd.DataFrame):
    levels = ecotype_levels(ecotypes["Ecotype"].tolist())
    code = r_match([str(e) for e in ecotypes["Ecotype"]], levels)
    code = np.where(code < 0, len(levels) + 1, code)
    eco = ecotypes.iloc[np.argsort(code, kind="stable")]
    return eco, levels


def _ttest_matrix(all_H: np.ndarray, rows_ids, eco, levels):
    p = np.ones((len(levels), all_H.shape[1]))
    for li, lvl in enumerate(levels):
        inc = set(eco.loc[eco["Ecotype"].astype(str) == lvl, "ID"].astype(str))
        mask = np.array([r in inc for r in rows_ids])
        for j in range(all_H.shape[1]):
            x = all_H[:, j]
            try:
                p[li, j] = welch_t_test_p(x[mask], x[~mask])
            except TTestError:
                p[li, j] = 1.0
    return p


def _ecotype_means(all_H, rows_ids, eco, levels, na_rm):
    H = np.empty((len(levels), all_H.shape[1]))
    for li, lvl in enumerate(levels):
        inc = set(eco.loc[eco["Ecotype"].astype(str) == lvl, "ID"].astype(str))
        mask = np.array([r in inc for r in rows_ids])
        H[li] = col_apply_mean(all_H[mask], na_rm=na_rm)
    return H


def _order_factor(values, levels):
    code = r_match([str(v) for v in values], levels)
    return np.argsort(np.where(code < 0, len(levels) + 10, code), kind="stable")


def recover_ecotypes(state_results: dict, model_ecotypes: pd.DataFrame, cell_types: list[str],
                     annotation: pd.DataFrame | None, *, r_exact: bool = True) -> EcotypeRecovery:
    """``ecotypes_recovery.R`` (bulk). ``state_results``: cell type -> StateRecovery."""
    eco, levels = _sorted_ecotypes(model_ecotypes)
    blocks = []
    rown = []
    assigned_ids: set = set()
    samples = None
    for ct in cell_types:
        r = state_results[ct]
        ab = r.abundances
        if samples is None:
            samples = list(ab.columns)
        blocks.append(_rt(ab.to_numpy(), r_exact))
        rown += [f"{ct}_{s}" for s in ab.index]
        states_in = set(eco.loc[eco["CellType"].astype(str) == ct, "State"].astype(str))
        assigned_ids |= {i for i, s in zip(r.assignment["ID"], r.assignment["State"]) if str(s) in states_in}
    allH = np.vstack(blocks)
    idx = r_match([str(i) for i in eco["ID"]], rown)
    all_H = np.full((len(idx), allH.shape[1]), np.nan)
    all_H[idx >= 0] = allH[idx[idx >= 0]]
    row_ids = [rown[k] if k >= 0 else "NA" for k in idx]
    comb = pd.DataFrame(all_H, index=row_ids, columns=samples)

    H = _ecotype_means(all_H, row_ids, eco, levels, na_rm=False)
    H = col_normalize(H)
    eco_ab = pd.DataFrame(H, index=levels, columns=samples)
    p = _ttest_matrix(all_H, row_ids, eco, levels)
    p_df = pd.DataFrame(p, index=levels, columns=samples)

    best = [which_max(H[:, j]) for j in range(H.shape[1])]
    clinical = pd.DataFrame({
        "ID": samples,
        "MaxEcotype": [levels[b] for b in best],
        "AssignmentP": [p[b, j] for j, b in enumerate(best)],
    })
    clinical["AssignmentQ"] = p_adjust_bh(clinical["AssignmentP"].to_numpy())
    clinical["AssignedToEcotypeStates"] = [s in assigned_ids for s in samples]
    clinical["Ecotype"] = [m if (q < 0.25 and a) else "Unassigned" for m, q, a in
                           zip(clinical["MaxEcotype"], clinical["AssignmentQ"], clinical["AssignedToEcotypeStates"])]
    tmp = read_clinical(samples, annotation)
    to_rem = [c for c in tmp.columns if c in clinical.columns and c != "ID"]
    tmp = tmp[[c for c in tmp.columns if c not in to_rem]]
    clinical = r_merge(clinical, tmp.reset_index(drop=True), "ID", all_x=True)
    clinical.index = pd.Index([str(i + 1) for i in range(len(clinical))], dtype=object)
    clinical = clinical.iloc[_order_factor(clinical["Ecotype"], levels + ["Unassigned"])]
    filt = clinical[clinical["Ecotype"] != "Unassigned"].copy()
    filt.index = pd.Index(filt["ID"].tolist(), dtype=object)

    order_cols = r_match(clinical["ID"].tolist(), samples)
    comb = comb.iloc[:, order_cols]
    return EcotypeRecovery(combined_state_abundances=pd.DataFrame(all_H, index=row_ids, columns=samples),
                           ecotype_abundance=eco_ab, assignment_p_vals=p_df, initial_assignment=clinical,
                           assignment=filt, ecotypes=eco)


def _table_frac(samples_of_cells, states_of_cells):
    """``as.data.frame(table(Sample, State))`` -> per sample fractions, as the
    State x Sample matrix produced by ``dcast(State ~ ID, value.var = "Frac")``."""
    s_lv = factor_levels(samples_of_cells)
    st_lv = factor_levels(states_of_cells)
    si = r_match(list(samples_of_cells), s_lv)
    ti = r_match(list(states_of_cells), st_lv)
    F = np.zeros((len(st_lv), len(s_lv)))
    for a, b in zip(ti, si):
        F[a, b] += 1
    frac = F / F.sum(axis=0, keepdims=True)  # Freq / sum(Freq): integer counts, exact
    return pd.DataFrame(frac, index=st_lv, columns=s_lv)


def recover_ecotypes_scrna(state_results: dict, model, cell_annotation: pd.DataFrame,
                           top_cols=("Ecotype",)) -> EcotypeRecovery:
    """``ecotypes_recovery_scRNA.R``: ecotype abundance per sample (column
    ``Sample`` of the cell annotation) from per-cell initial state assignments."""
    eco_all, levels_all = _sorted_ecotypes(model.ecotypes)
    blocks = []
    rown = []
    col_names: list = []
    classes_by_ct = {}
    cell_ids: list = []
    for ct in model.cell_type_order:
        if ct not in state_results:
            continue
        r = state_results[ct]
        ctm = model.cell_types[ct]
        mp_init = [str(s) for s in ctm.mapping["InitialState"]]
        mp_state = [str(s) for s in ctm.mapping["State"]]
        ia = r.initial_assignment
        clin = read_clinical(ia["ID"].tolist(), cell_annotation)
        samp = [None if pd.isna(v) else str(v) for v in clin["Sample"]]
        for i in ia["ID"]:
            if i not in cell_ids:
                cell_ids.append(i)
        ok = [s is not None for s in samp]
        Hraw = _table_frac([s for s, k in zip(samp, ok) if k],
                           [str(x) for x, k in zip(ia["State"], ok) if k])
        ridx = r_match(mp_init, list(Hraw.index))
        Hm = np.full((len(mp_init), Hraw.shape[1]), np.nan)
        Hm[ridx >= 0] = Hraw.to_numpy()[ridx[ridx >= 0]]
        blocks.append((Hm, list(Hraw.columns)))
        rown += [f"{ct}_{s}" for s in mp_state]
        for c in Hraw.columns:
            if c not in col_names:
                col_names.append(c)
        # dominant initial state per sample, mapped to states that are in ecotypes
        HR = Hraw.to_numpy()
        dom = [Hraw.index[which_max(HR[:, j])] for j in range(HR.shape[1])]
        states_in = set(eco_all.loc[eco_all["CellType"].astype(str) == ct, "State"].astype(str))
        ids = set()
        for sid, d in zip(Hraw.columns, dom):
            k = mp_init.index(d) if d in mp_init else -1
            if k >= 0 and mp_state[k] in states_in:
                ids.add(sid)
        classes_by_ct[ct] = ids
    # rbind.fill
    allH = np.full((len(rown), len(col_names)), np.nan)
    r0 = 0
    for Hm, cols in blocks:
        ci = r_match(cols, col_names)
        allH[r0:r0 + Hm.shape[0], ci] = Hm
        r0 += Hm.shape[0]
    eco = eco_all[eco_all["CellType"].astype(str).isin(list(classes_by_ct))]
    levels = levels_all
    idx = r_match([str(i) for i in eco["ID"]], rown)
    all_H = np.full((len(idx), allH.shape[1]), np.nan)
    all_H[idx >= 0] = allH[idx[idx >= 0]]
    row_ids = [rown[k] if k >= 0 else "NA" for k in idx]
    samples = col_names

    H = _ecotype_means(all_H, row_ids, eco, levels, na_rm=True)
    H = col_normalize(H, na_rm=True)
    p = _ttest_matrix(all_H, row_ids, eco, levels)
    best = [which_max(H[:, j]) for j in range(H.shape[1])]
    assigned = set().union(*classes_by_ct.values()) if classes_by_ct else set()
    clinical = pd.DataFrame({
        "ID": samples,
        "MaxEcotype": [levels[b] if b >= 0 else "Unassigned" for b in best],
        "AssignmentP": [p[b, j] if b >= 0 else np.nan for j, b in enumerate(best)],
    })
    clinical["AssignmentQ"] = p_adjust_bh(clinical["AssignmentP"].to_numpy())
    clinical["AssignedToEcotypeStates"] = [s in assigned for s in samples]
    clinical["Ecotype"] = [m if a else "Unassigned" for m, a in zip(clinical["MaxEcotype"], clinical["AssignedToEcotypeStates"])]
    ann = read_clinical(cell_ids, cell_annotation)
    for x in [c for c in top_cols if c in ann.columns]:
        pairs = {}
        dup = False
        for s, v in zip(ann["Sample"], ann[x]):
            if s is None:
                continue
            pairs.setdefault(str(s), set()).add(v)
        if any(len(v) > 1 for v in pairs.values()):
            dup = True
        if not dup:
            clinical[x] = [None if s not in pairs else str(next(iter(pairs[s]))) for s in clinical["ID"]]
    clinical.index = pd.Index([str(i + 1) for i in range(len(clinical))], dtype=object)
    clinical = clinical.iloc[_order_factor(clinical["Ecotype"], levels + ["Unassigned"])]
    filt = clinical[clinical["Ecotype"] != "Unassigned"].copy()
    return EcotypeRecovery(combined_state_abundances=pd.DataFrame(all_H, index=row_ids, columns=samples),
                           ecotype_abundance=pd.DataFrame(H, index=levels, columns=samples),
                           assignment_p_vals=pd.DataFrame(p, index=levels, columns=samples),
                           initial_assignment=clinical, assignment=filt, ecotypes=eco)


# ----------------------------------------------------------------------------
# spatial (Visium)
# ----------------------------------------------------------------------------
def spatial_states(state_results: dict, cell_types: list[str], fractions: pd.DataFrame,
                   coords: pd.DataFrame, models: dict, malignant_cell: str | None,
                   sample_name: str, *, r_exact: bool = True):
    """``spatial_states.R``: per-spot state abundances weighted by CSx fractions.

    Returns ``(long, wide)`` as ``state_abundances_long.txt`` / ``state_abundances.txt``.
    """
    frac = fractions.copy()
    frac.columns = make_names(list(frac.columns))
    if malignant_cell is None or malignant_cell == "NULL" or malignant_cell not in frac.columns:
        malignant_cell = "Placeholder"
        frac["Placeholder"] = 1.0
    id_col = frac.columns[0]
    parts = []
    for ct in cell_types:
        ctm = models[ct]
        H = _rt(state_results[ct].initial_abundances.to_numpy(), r_exact)
        rows0 = list(state_results[ct].initial_abundances.index)
        spots = list(state_results[ct].initial_abundances.columns)
        B = (H == H.max(axis=0, keepdims=True)).astype(np.float64)
        mp_state = [str(s) for s in ctm.mapping.iloc[:, 0]]
        mp_init = [str(s) for s in ctm.mapping.iloc[:, 1]]
        ridx = r_match(mp_init, rows0)
        Bm = np.full((len(ridx), B.shape[1]), np.nan)
        Bm[ridx >= 0] = B[ridx[ridx >= 0]]
        Hdf = pd.DataFrame(Bm.T, columns=mp_state)
        Hdf.insert(0, "Row.names", spots)
        tmp = pd.DataFrame({id_col: frac[id_col].to_numpy(dtype=object),
                            ct: frac[ct].to_numpy(dtype=np.float64),
                            "Malignant": frac[malignant_cell].to_numpy(dtype=np.float64)})
        tmp = tmp.rename(columns={id_col: "Row.names"})
        merged = r_merge(Hdf, tmp, "Row.names", all_x=True)
        merged = merged.rename(columns={"Row.names": "ID"})
        # melt(id.vars = c(ID, ct, Malignant)): state columns stacked in order
        for s in mp_state:
            d = pd.DataFrame({"ID": merged["ID"].to_numpy(), "CSx_fraction": merged[ct].to_numpy(),
                              "Malignant": merged["Malignant"].to_numpy(), "State": s,
                              "NMF": merged[s].to_numpy(dtype=np.float64)})
            parts.append((ct, d))
        # per-cell-type normalisation over all its states
    out = []
    for ct in cell_types:
        ds = [d for c, d in parts if c == ct]
        m = pd.concat(ds, ignore_index=True)
        m["CellType"] = ct
        m["Sample"] = sample_name
        m["NMF_weighted_raw"] = m["NMF"].to_numpy() * m["CSx_fraction"].to_numpy()
        q = quantile7(m["NMF_weighted_raw"].to_numpy(), 0.99, na_rm=True)
        ab = m["NMF_weighted_raw"].to_numpy() / q
        m["Abundance"] = np.where(ab > 1, 1.0, ab)
        out.append(m)
    long = pd.concat(out, ignore_index=True)
    ci = r_match(long["ID"].tolist(), coords["ID"].tolist())
    long["X"] = [coords["X"].iloc[k] if k >= 0 else np.nan for k in ci]
    long["Y"] = [coords["Y"].iloc[k] if k >= 0 else np.nan for k in ci]
    long["State_ID"] = long["CellType"] + "_" + long["State"]
    # dcast(ID + X + Y + Sample + Malignant ~ State_ID, value.var = "Abundance")
    keys = ["ID", "X", "Y", "Sample", "Malignant"]
    kdf = long[keys].drop_duplicates()
    o = r_order(*[kdf[k].tolist() for k in keys])
    kdf = kdf.iloc[o].reset_index(drop=True)
    sid_levels = factor_levels(long["State_ID"].tolist())
    keytuples = list(map(tuple, kdf.astype(object).itertuples(index=False, name=None)))
    pos = {kt: i for i, kt in enumerate(keytuples)}
    W = np.full((len(kdf), len(sid_levels)), np.nan)
    sidx = {s: i for i, s in enumerate(sid_levels)}
    for kt, sid, v in zip(map(tuple, long[keys].astype(object).itertuples(index=False, name=None)),
                          long["State_ID"], long["Abundance"]):
        W[pos[kt], sidx[sid]] = v
    wide = pd.concat([kdf, pd.DataFrame(W, columns=sid_levels)], axis=1)
    long = long.drop(columns=["State_ID"])
    return long, wide


def spatial_ecotypes(wide: pd.DataFrame, ecotypes: pd.DataFrame, *, r_exact: bool = True) -> pd.DataFrame:
    """``spatial_ecotypes.R``: per-spot ecotype abundances (``ecotype_abundances.txt``)."""
    wide = wide.copy()
    wide.columns = make_names(list(wide.columns), unique=True)
    num_cols = [c for c in wide.columns[5:]]
    vals = _rt(wide[num_cols].to_numpy(dtype=np.float64), r_exact)
    mal = _rt(wide["Malignant"].to_numpy(dtype=np.float64), r_exact)
    ecos = [str(e) for e in ecotypes["Ecotype"]]
    if all(f"CE{k}" in ecos for k in range(1, 11)):
        levels = [f"CE{k}" for k in range(1, 11)]
    else:
        levels = factor_levels(ecos)
    o = _order_factor(ecos, levels)
    mp = ecotypes.iloc[o]
    mp_ids = [str(i) for i in mp["ID"]]
    mp_eco = np.array([str(e) for e in mp["Ecotype"]], dtype=object)
    out = []
    samples = wide["Sample"].astype(str).tolist()
    for spl in factor_levels(samples):
        rows = np.array([s == spl for s in samples])
        ci = r_match(mp_ids, num_cols)
        tmp = np.full((rows.sum(), len(ci)), np.nan)
        sub = vals[rows]
        tmp[:, ci >= 0] = sub[:, ci[ci >= 0]]
        tmp[np.isnan(tmp)] = 0.0
        tmp[tmp > 1] = 1.0
        E = np.column_stack([apply_mean_rows(tmp[:, mp_eco == lv]) for lv in levels])
        E[np.isnan(E)] = 0.0
        q = quantile7(E.ravel(order="F"), 0.99, na_rm=True)
        E = E / q
        E[E > 1] = 1.0
        df = wide.loc[rows, wide.columns[:5]].copy()
        # do.call(rbind, <named list of splits>) row names: "<Sample>.<row number>"
        df.index = pd.Index([f"{spl}.{k + 1}" for k in np.flatnonzero(rows)], dtype=object)
        m = mal[rows].copy()
        m[np.isnan(m)] = 0.0
        m = m - quantile7(m, 0.01, na_rm=True)
        m = np.where(m < 0, 0.0, m)
        m = m / quantile7(m, 0.99, na_rm=True)
        m = np.where(m > 1, 1.0, m)
        df["Malignant"] = m
        for k, lv in enumerate(levels):
            df[lv] = E[:, k]
        out.append(df)
    return pd.concat(out)
