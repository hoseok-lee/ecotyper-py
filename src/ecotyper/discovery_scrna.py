"""scRNA-seq discovery (``EcoTyper_discovery_scRNA.R``):

1. cell-type-specific genes (``state_discovery_scRNA_filter_genes.R``);
2. cell-cell correlation matrices (``state_discovery_scRNA_distances.R``) and
   NMF on them (``state_discovery_NMF.R`` with ``discovery_cross_cor``);
3. rank selection;
4. initial states (``state_discovery_initial_plots_scRNA.R``);
5. marker genes and NMF re-discovery on expression
   (``state_discovery_extract_features_scRNA.R``);
6. initial states on expression;
7. QC filter (``state_discovery_first_filter_scRNA.R``);
8. ecotypes (``ecotypes_scRNA.R``, ``ecotypes_assign_samples_scRNA.R``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import discovery_steps as dsc
from .model import ecotype_levels
from .rcompat.frame import factor_levels, r_match, r_order
from .rcompat.rng import RRandom
from .rcompat.stats import (TTestError, apply_mean_rows, col_normalize, cor_matrix, log2p1, p_adjust_bh,
                            phyper_upper, rank_average, row_var, row_wilcoxon_twosample, which_max,
                            welch_t_test_p)
from .recovery_steps import EcotypeRecovery, _ecotype_means, _sorted_ecotypes, _table_frac, read_clinical


def _rt(x, r_exact=True):
    return dsc._rt(x, r_exact)


# ----------------------------------------------------------------------------
# 1. cell type specific genes
# ----------------------------------------------------------------------------
def do_de(data: pd.DataFrame, variable, value: str) -> pd.DataFrame:
    """``doDE(data, variable, value)`` (lib/misc.R): Wilcoxon rank-sum test of
    one group against all other samples."""
    variable = np.array([str(v) for v in variable], dtype=object)
    X = data.to_numpy(dtype=np.float64)
    inc = variable == value
    fc = apply_mean_rows(X[:, inc], na_rm=True) - apply_mean_rows(X[:, ~inc], na_rm=True)
    p = row_wilcoxon_twosample(X[:, inc], X[:, ~inc])
    q = p_adjust_bh(p)
    df = pd.DataFrame({"Gene": [str(g) for g in data.index], "State": value, "FC": fc, "P": p, "Q": q})
    df["P"] = np.where(np.isnan(df["P"]), 1.0, df["P"])
    df["Q"] = np.where(np.isnan(df["Q"]), 1.0, df["Q"])
    o = r_order(df["State"].tolist(), (-df["FC"]).tolist(), df["P"].tolist())
    out = df.iloc[o]
    out.index = pd.Index([str(i + 1) for i in o], dtype=object)
    return out


def do_cv(data: pd.DataFrame, variable, value: str) -> pd.DataFrame:
    """``doCV`` (lib/misc.R): within-group variance, decreasing."""
    variable = np.array([str(v) for v in variable], dtype=object)
    X = data.to_numpy(dtype=np.float64)[:, variable == value]
    v = row_var(X)
    v = np.where(np.isnan(v), 0.0, v)
    df = pd.DataFrame({"Gene": [str(g) for g in data.index], "State": value, "Var": v})
    o = r_order(df["State"].tolist(), (-df["Var"]).tolist())
    return df.iloc[o]


def de_subsample(annotation: pd.DataFrame, cell_ids) -> pd.DataFrame:
    """Annotation of the cells used for the DE test: ``set.seed(1234)`` then up
    to 500 cells per cell type, cell types in sorted order."""
    ann = annotation[annotation["ID"].astype(str).isin(set(map(str, cell_ids)))]
    rng = RRandom(1234)
    parts = []
    cts = [str(c) for c in ann["CellType"]]
    for ct in factor_levels(cts):
        spl = ann[[c == ct for c in cts]]
        k = min(500, len(spl))
        idx = rng.sample_int(len(spl), k) - 1
        parts.append(spl.iloc[idx])
    return pd.concat(parts)


@dataclass
class SpecificGenes:
    raw: pd.DataFrame
    filtered: pd.DataFrame


def cell_type_specific_genes(expr: pd.DataFrame, annotation: pd.DataFrame, cell_types, mode="cell type specific",
                             scale_column=None) -> dict:
    """Step 1 for all cell types (each R process repeats the same subsampling)."""
    ann_in = annotation[annotation["ID"].astype(str).isin(set(map(str, expr.columns)))]
    counts = {ct: int((ann_in["CellType"].astype(str) == ct).sum()) for ct in cell_types}
    sub = de_subsample(ann_in, expr.columns)
    keep = set(sub["ID"].astype(str))
    cols = [c for c in expr.columns if str(c) in keep]
    log_data = pd.DataFrame(log2p1(expr[cols].to_numpy(dtype=np.float64)), index=expr.index, columns=cols)
    clinical = read_clinical(cols, annotation)
    by = None if scale_column in (None, "NULL") else [str(v) for v in clinical[scale_column]]
    out = {}
    scaled = None
    for ct in cell_types:
        if counts.get(ct, 0) < 50:
            continue
        if mode == "cell type specific":
            if scaled is None:
                scaled = dsc.scale_data(log_data, by=by)
                scaled = pd.DataFrame(np.nan_to_num(scaled.to_numpy(), nan=0.0), index=scaled.index,
                                      columns=scaled.columns)
            gi = do_de(scaled, clinical["CellType"].tolist(), ct)
            gi = gi.rename(columns={"State": "CellType"})
            filt = gi[(gi["Q"] <= 0.05) & (gi["FC"] > 0)]
        else:
            n_genes = int(str(mode).replace("Top_", ""))
            gi = do_cv(log_data, clinical["CellType"].tolist(), ct).rename(columns={"State": "CellType"})
            filt = gi.iloc[: min(len(gi), n_genes)]
        out[ct] = SpecificGenes(gi, filt)
    return out


# ----------------------------------------------------------------------------
# 2. cell-cell correlations
# ----------------------------------------------------------------------------
def fast_cor(X: np.ndarray) -> np.ndarray:
    """``HiClimR::fastCor(X)`` with its defaults (``nSplit = 1``, ``optBLAS = FALSE``):
    ``cor()`` between columns, zero-variance columns set to NA."""
    v = row_var(np.ascontiguousarray(X.T))
    ii = np.flatnonzero(v > 0)
    r = cor_matrix(X[:, ii])
    out = np.full((X.shape[1], X.shape[1]), np.nan)
    out[np.ix_(ii, ii)] = r
    return out


@dataclass
class CrossCor:
    cell_type: str
    log_data: pd.DataFrame  # genes x cells (all genes)
    scaled_filt: pd.DataFrame  # specific genes x cells
    cor: pd.DataFrame  # cells x cells
    scaled_full: pd.DataFrame | None = None  # all genes x cells, scaled (expression_full_matrix_scaled.txt)


def cross_correlation(expr: pd.DataFrame, annotation: pd.DataFrame, cell_type: str, specific: list | None,
                      scale_column=None) -> CrossCor:
    ann = annotation[annotation["CellType"].astype(str) == cell_type]
    if len(ann) < 50:
        raise ValueError(f"Only {len(ann)} single cells are available for cell type: {cell_type}")
    if len(ann) > 2500:
        rng = RRandom(1234)
        ann = ann.iloc[rng.sample_int(len(ann), 2500) - 1]
    keep = set(ann["ID"].astype(str))
    cols = [c for c in expr.columns if str(c) in keep]
    log_data = pd.DataFrame(log2p1(expr[cols].to_numpy(dtype=np.float64)), index=expr.index, columns=cols)
    clinical = read_clinical(cols, annotation)
    by = None if scale_column in (None, "NULL") else [str(v) for v in clinical[scale_column]]
    scaled = dsc.scale_data(log_data, by=by)
    S = np.nan_to_num(scaled.to_numpy(), nan=0.0)
    scaled = pd.DataFrame(S, index=scaled.index, columns=cols)
    scaled_full = scaled
    if specific is not None:
        ks = set(map(str, specific))
        scaled = scaled[[str(g) in ks for g in scaled.index]]
    C = fast_cor(scaled.to_numpy())
    return CrossCor(cell_type, log_data, scaled, pd.DataFrame(C, index=cols, columns=cols), scaled_full)


# ----------------------------------------------------------------------------
# 5. marker genes for the re-discovery on expression
# ----------------------------------------------------------------------------
@dataclass
class ScFeatures:
    cell_type: str
    full_fcs: pd.DataFrame
    top_fcs: pd.DataFrame
    log_full: pd.DataFrame
    log_top: pd.DataFrame
    top_scaled: pd.DataFrame
    full_scaled: pd.DataFrame


def extract_features_scrna(cc: CrossCor, initial_assignment: pd.DataFrame, r_exact=True) -> ScFeatures:
    scaled = dsc._rt_df(cc.scaled_filt, r_exact)
    logd = cc.log_data.iloc[r_match([str(g) for g in scaled.index], [str(g) for g in cc.log_data.index])]
    logd = dsc._rt_df(logd, r_exact)
    if scaled.shape[0] < 50:
        raise ValueError(f"Only {scaled.shape[0]} genes were imputed for cell type: {cc.cell_type}")
    ia = initial_assignment
    ann = ia.iloc[r_match([str(c) for c in scaled.columns], ia["ID"].astype(str).tolist())]
    fcs = dsc.get_fc_for_clusters(scaled, ann["State"].tolist())
    parts = []
    st = fcs["State"].tolist()
    for lv in factor_levels([s for s in st if s is not None]):
        spl = fcs[[s == lv for s in st]].copy()
        spl["Rank"] = rank_average(-spl["MaxFC"].to_numpy())
        spl.index = pd.Index([f"{lv}.{i}" for i in spl.index], dtype=object)
        parts.append(spl)
    fcs = pd.concat(parts)
    fcs = fcs.iloc[r_order(fcs["Rank"].tolist(), (-fcs["MaxFC"]).tolist())]
    top = fcs[fcs["MaxFC"] > 0]
    top = top.iloc[: min(len(top), 1000)]
    genes = top["Gene"].tolist()
    log_top = logd.iloc[r_match(genes, [str(g) for g in logd.index])]
    top_scaled = scaled.iloc[r_match(genes, [str(g) for g in scaled.index])]
    return ScFeatures(cc.cell_type, fcs, top, logd, log_top, top_scaled, scaled)


# ----------------------------------------------------------------------------
# 8. ecotypes
# ----------------------------------------------------------------------------
def discover_ecotypes_scrna(initial: dict, states: dict, cell_types, annotation: pd.DataFrame,
                            p_value_cutoff: float = 1.0, min_states: int = 3) -> dsc.Ecotypes:
    maps = []
    rows = {}  # "<ct>_<state>" -> {sample: Max}
    samples_all: list = []
    for ct in cell_types:
        m = states[ct].mapping[["State", "InitialState"]].copy()
        m["CellType"] = ct
        maps.append(m)
        ia = initial[ct].initial_assignment
        clin = read_clinical(ia["ID"].tolist(), annotation)
        samp = [None if pd.isna(v) else str(v) for v in clin["Sample"]]
        ok = [s is not None for s in samp]
        F = _table_counts([s for s, k in zip(samp, ok) if k], [str(x) for x, k in zip(ia["State"], ok) if k])
        for sid in F.columns:
            col = F[sid].to_numpy()
            mx = col.max() if len(col) else np.nan
            for st_, v in zip(F.index, col):
                rows.setdefault(f"{ct}_{st_}", {})[sid] = 1.0 if (mx > 0 and v == mx) else 0.0
        samples_all += list(F.columns)
    am = pd.concat(maps, ignore_index=True)
    am["InitialID"] = am["CellType"] + "_" + am["InitialState"]
    am["ID"] = am["CellType"] + "_" + am["State"]
    samples = factor_levels(list(dict.fromkeys(samples_all)))
    B = np.zeros((len(am), len(samples)))
    for i, iid in enumerate(am["InitialID"]):
        d = rows.get(iid)
        if d is None:
            B[i] = np.nan
            continue
        for j, s in enumerate(samples):
            B[i, j] = d.get(s, 0.0)
    sid = am["ID"].tolist()
    binary = pd.DataFrame(B, index=sid, columns=samples)
    n = len(sid)
    N = len(samples)
    J = np.full((n, n), np.nan)
    Bb = B > 0
    sums = np.nansum(B, axis=1)
    for i in range(n):
        for j in range(i, n):
            inter = float((Bb[i] & Bb[j]).sum())
            idx = inter / (sums[i] + sums[j] - inter)
            p = phyper_upper(inter, sums[i], N - sums[i], sums[j])
            if np.isnan(p) or p >= p_value_cutoff:
                idx = 0.0
            J[i, j] = J[j, i] = idx
    J[np.isnan(J)] = 0.0
    jac = pd.DataFrame(J, index=sid, columns=sid)
    D = 1.0 - J
    d = np.concatenate([D[j + 1:, j] for j in range(n - 1)])
    from .rcompat.cluster import cutree, hclust, silhouette_avg_width_dist

    hc = hclust(d, n, "average")
    sils = [silhouette_avg_width_dist(cutree(hc, k), d) for k in range(2, n)]
    k_opt = 2 + which_max(np.array(sils))
    clust = cutree(hc, k_opt)
    sil0 = silhouette_avg_width_dist(clust, d)
    parts = [s.rsplit("_", 1) for s in sid]
    top = pd.DataFrame({"CellType": [p[0] for p in parts], "State": [p[1] for p in parts],
                        "InitialEcotype": [f"IE{c:02d}" for c in clust]}, index=pd.Index(sid, dtype=object))
    init_eco = top.copy()
    top["ID"] = list(top.index)
    lv = factor_levels(top["InitialEcotype"].tolist())
    top = top.iloc[np.argsort(r_match(top["InitialEcotype"].tolist(), lv), kind="stable")]
    tb = {x: int((top["InitialEcotype"] == x).sum()) for x in lv}
    keep = {x for x, c in tb.items() if c >= min_states}
    if len(keep) < 2:
        keep = {x for x, c in tb.items() if c >= min_states - 1}
    top = top[[x in keep for x in top["InitialEcotype"]]].copy()
    nm = list(dict.fromkeys(top["InitialEcotype"].tolist()))
    emap = {x: f"E{k + 1}" for k, x in enumerate(nm)}
    top["Ecotype"] = [emap[x] for x in top["InitialEcotype"]]
    lvls = ecotype_levels(top["Ecotype"].tolist())
    top = top.iloc[np.argsort(r_match(top["Ecotype"].tolist(), lvls), kind="stable")]
    sel = r_match(top["ID"].tolist(), sid)
    D2 = D[np.ix_(sel, sel)]
    d2 = np.concatenate([D2[j + 1:, j] for j in range(len(sel) - 1)]) if len(sel) > 1 else np.empty(0)
    sil = silhouette_avg_width_dist(np.array([int(e[1:]) for e in top["Ecotype"]]), d2) if len(sel) > 1 else np.nan
    return dsc.Ecotypes(am[["State", "InitialState", "CellType", "InitialID", "ID"]], binary, jac, sil0,
                        init_eco, top.reset_index(drop=True), sil, int(k_opt))


def _table_counts(samples, states) -> pd.DataFrame:
    s_lv = factor_levels(samples)
    t_lv = factor_levels(states)
    F = np.zeros((len(t_lv), len(s_lv)))
    si = r_match(list(samples), s_lv)
    ti = r_match(list(states), t_lv)
    for a, b in zip(ti, si):
        F[a, b] += 1
    return pd.DataFrame(F, index=t_lv, columns=s_lv)


def assign_samples_scrna(initial: dict, states: dict, eco: pd.DataFrame, cell_types, annotation: pd.DataFrame,
                         top_cols=("Ecotype",)) -> EcotypeRecovery:
    """``ecotypes_assign_samples_scRNA.R``."""
    eco_s, levels = _sorted_ecotypes(eco)
    blocks, rown, col_names = [], [], []
    assigned: set = set()
    cell_ids: list = []
    for ct in cell_types:
        mp_init = [str(s) for s in states[ct].mapping["InitialState"]]
        mp_state = [str(s) for s in states[ct].mapping["State"]]
        ia = initial[ct].initial_assignment
        clin = read_clinical(ia["ID"].tolist(), annotation)
        samp = [None if pd.isna(v) else str(v) for v in clin["Sample"]]
        for i in ia["ID"]:
            if i not in cell_ids:
                cell_ids.append(i)
        ok = [s is not None for s in samp]
        Hraw = _table_frac([s for s, k in zip(samp, ok) if k], [str(x) for x, k in zip(ia["State"], ok) if k])
        ridx = r_match(mp_init, list(Hraw.index))
        Hm = np.full((len(mp_init), Hraw.shape[1]), np.nan)
        Hm[ridx >= 0] = Hraw.to_numpy()[ridx[ridx >= 0]]
        blocks.append((Hm, list(Hraw.columns)))
        rown += [f"{ct}_{s}" for s in mp_state]
        for c in Hraw.columns:
            if c not in col_names:
                col_names.append(c)
        HR = Hraw.to_numpy()
        st_in = set(eco_s.loc[eco_s["CellType"].astype(str) == ct, "State"].astype(str))
        for j, sid in enumerate(Hraw.columns):
            k = which_max(HR[:, j])
            d = Hraw.index[k] if k >= 0 else "Unassigned"
            if d in mp_init and mp_state[mp_init.index(d)] in st_in:
                assigned.add(sid)
    allH = np.full((len(rown), len(col_names)), np.nan)
    r0 = 0
    for Hm, cols in blocks:
        allH[r0:r0 + Hm.shape[0], r_match(cols, col_names)] = Hm
        r0 += Hm.shape[0]
    idx = r_match([str(i) for i in eco_s["ID"]], rown)
    all_H = np.full((len(idx), allH.shape[1]), np.nan)
    all_H[idx >= 0] = allH[idx[idx >= 0]]
    row_ids = [rown[k] if k >= 0 else "NA" for k in idx]
    H = col_normalize(_ecotype_means(all_H, row_ids, eco_s, levels, na_rm=True), na_rm=True)
    p = np.full((len(levels), all_H.shape[1]), np.nan)
    for li, lvl in enumerate(levels):
        inc = set(eco_s.loc[eco_s["Ecotype"].astype(str) == lvl, "ID"].astype(str))
        mask = np.array([r in inc for r in row_ids])
        for j in range(all_H.shape[1]):
            x = all_H[:, j]
            try:
                p[li, j] = welch_t_test_p(x[mask], x[~mask])
            except TTestError:
                p[li, j] = np.nan
    best = [which_max(H[:, j]) for j in range(H.shape[1])]
    clinical = pd.DataFrame({"ID": col_names, "MaxEcotype": [levels[b] if b >= 0 else "Unassigned" for b in best],
                             "AssignmentP": [p[b, j] if b >= 0 else np.nan for j, b in enumerate(best)]})
    clinical["AssignmentQ"] = p_adjust_bh(clinical["AssignmentP"].to_numpy())
    clinical["AssignedToEcotypeStates"] = [s in assigned for s in col_names]
    clinical["Ecotype"] = [m if a else "Unassigned" for m, a in zip(clinical["MaxEcotype"],
                                                                    clinical["AssignedToEcotypeStates"])]
    ann = read_clinical(cell_ids, annotation)
    for x in [c for c in top_cols if c in ann.columns]:
        pairs: dict = {}
        for s, v in zip(ann["Sample"], ann[x]):
            if not pd.isna(s):
                pairs.setdefault(str(s), set()).add(v)
        if not any(len(v) > 1 for v in pairs.values()):
            clinical[x] = [None if s not in pairs else str(next(iter(pairs[s]))) for s in clinical["ID"]]
    clinical.index = pd.Index([str(i + 1) for i in range(len(clinical))], dtype=object)
    lv2 = levels + ["Unassigned"]
    clinical = clinical.iloc[np.argsort(r_match(clinical["Ecotype"].tolist(), lv2), kind="stable")]
    filt = clinical[clinical["Ecotype"] != "Unassigned"].copy()
    filt.index = pd.Index(filt["ID"].tolist(), dtype=object)
    filt = filt.iloc[np.argsort(r_match(filt["Ecotype"].tolist(), levels), kind="stable")]
    return EcotypeRecovery(combined_state_abundances=pd.DataFrame(all_H, index=row_ids, columns=col_names),
                           ecotype_abundance=pd.DataFrame(H, index=levels, columns=col_names),
                           assignment_p_vals=pd.DataFrame(p, index=levels, columns=col_names),
                           initial_assignment=clinical, assignment=filt, ecotypes=eco_s)
