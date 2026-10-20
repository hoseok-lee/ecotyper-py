"""Cell state and ecotype discovery from purified (CIBERSORTx HiRes) or
presorted profiles: ``state_discovery_*.R``, ``ecotypes.R``,
``ecotypes_assign_samples.R`` (bulk and presorted discovery wrappers).

Each R script becomes one function; values that R writes to a text file and
the next script reads back are passed through :func:`_rt` (``r_exact``) so the
port sees exactly the numbers R sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model import ecotype_levels
from .nmf import (NMFFit, cophcor, dispersion, merge_runs, nmf_brunet, posneg, predict_columns,
                  silhouettes)
from .rcompat.cluster import cutree, dist_euclidean_rows, hclust, silhouette_avg_width_dist
from .rcompat.frame import factor_levels, r_match, r_merge, r_order
from .rcompat.io import format_real
from .rcompat.parse import text_roundtrip
from .rcompat.stats import (TTestError, apply_mean_rows, col_apply_mean, col_normalize, log2p1,
                            p_adjust_bh, phyper_upper, quantile7, r_mad, r_median, r_sum, row_means,
                            row_var, scale_rows, which_max, welch_t_test_p)
from .recovery_steps import read_clinical


def _rt(x, r_exact=True):
    return text_roundtrip(x, "r") if r_exact else np.asarray(x, dtype=np.float64)


def _rt_df(df: pd.DataFrame, r_exact=True) -> pd.DataFrame:
    return pd.DataFrame(_rt(df.to_numpy(dtype=np.float64), r_exact), index=df.index, columns=df.columns)


def _log2_plain(X):
    X = np.ascontiguousarray(X, dtype=np.float64)
    out = np.empty_like(X)
    flat = X.ravel()
    o = out.ravel()
    for i in range(flat.size):
        v = flat[i]
        o[i] = math.log2(v) if v == v else np.nan
    return out


def log_transform(X: np.ndarray) -> np.ndarray:
    """``log2(x)`` if min >= 1, ``log2(x + 1)`` if min >= 0, else unchanged."""
    mn = np.nanmin(X)
    if mn >= 1:
        return _log2_plain(X)
    if mn >= 0:
        return log2p1(X)
    return X.copy()


# ----------------------------------------------------------------------------
# step: extract features (state_discovery_extract_features.R)
# ----------------------------------------------------------------------------
def get_variable_genes(log_data: pd.DataFrame) -> pd.DataFrame:
    """``get_variable_genes`` (lib/misc.R): top 1000 genes by normalised
    dispersion within mean-expression bins, ordered by decreasing dispersion."""
    if log_data.shape[0] < 1000:
        return log_data
    X = log_data.to_numpy(dtype=np.float64)
    mean = row_means(X)
    var = row_var(X)
    disp = var / mean
    probs = 0.1 + np.arange(19) * 0.05  # seq(0.1, 1, 0.05)
    q = quantile7(mean, probs)
    uq = list(dict.fromkeys(q.tolist()))
    breaks = np.sort(np.array([-np.inf] + uq + [np.inf]))
    # .bincode(right = TRUE): breaks[i] < x <= breaks[i + 1]
    code = np.searchsorted(breaks, mean, side="left")
    code = np.where(np.isnan(mean), -1, code)
    med = {}
    mad = {}
    for c in np.unique(code):
        d = disp[code == c]
        if np.isnan(d).any():
            med[c] = np.nan
            mad[c] = np.nan
        else:
            med[c] = r_median(d)
            mad[c] = r_mad(d)
    bmed = np.array([med[c] for c in code])
    bmad = np.array([mad[c] for c in code])
    with np.errstate(divide="ignore", invalid="ignore"):
        dn = np.abs(disp - bmed) / bmad
    n_genes = min(int((~np.isnan(dn)).sum()), 1000)
    if n_genes == 0:
        return None
    key = np.where(np.isnan(dn), 0.0, -dn)
    o = np.lexsort((key, np.isnan(dn)))
    return log_data.iloc[o[:n_genes]]


def scale_data(data: pd.DataFrame, by=None) -> pd.DataFrame:
    """``scale_data`` (lib/misc.R): row z-scores within groups of samples.

    Note: the R pipeline passes the *name* of the annotation column (a single
    string) as ``by``, so every sample falls in the same group; pass a scalar
    to reproduce that, or one label per sample for true within-group scaling."""
    cols = list(data.columns)
    if by is None or np.ndim(by) == 0:
        variable = ["all samples" if by is None else str(by)] * len(cols)
    else:
        variable = [str(v) for v in by]
    X = data.to_numpy(dtype=np.float64)
    parts = []
    names = []
    for lv in factor_levels(variable):
        ids = [c for c, v in zip(cols, variable) if v == lv]
        if len(ids) < 2:
            continue
        idx = [i for i, c in enumerate(cols) if c in set(ids)]
        parts.append(scale_rows(X[:, idx]))
        names += [cols[i] for i in idx]
    S = np.hstack(parts)
    m = r_match(cols, names)
    out = np.full(X.shape, np.nan)
    ok = m >= 0
    out[:, ok] = S[:, m[ok]]
    return pd.DataFrame(out, index=data.index, columns=cols)


@dataclass
class Features:
    cell_type: str
    log_data: pd.DataFrame
    top_log2: pd.DataFrame  # expression_full_matrix_log2.txt (variable genes, log2)
    top_scaled: pd.DataFrame  # expression_top_genes_scaled.txt
    full_scaled: pd.DataFrame  # expression_full_matrix_scaled.txt


def extract_features(raw_input: pd.DataFrame, cell_type: str, annotation: pd.DataFrame | None = None,
                     scale_column: str | None = None, specific_genes: list | None = None) -> Features | None:
    """``state_discovery_extract_features.R``. ``raw_input``: genes x samples
    (``read.delim(..., row.names = 1)``). Returns None when fewer than 50 genes."""
    if raw_input.shape[0] < 50:
        return None
    X = log_transform(raw_input.to_numpy(dtype=np.float64))
    log_data = pd.DataFrame(X, index=raw_input.index, columns=raw_input.columns)
    if specific_genes is not None:
        keep = set(specific_genes)
        log_data = log_data[[g in keep for g in log_data.index]]
        if log_data.shape[0] < 50:
            return None
    data = get_variable_genes(log_data)
    by = None if scale_column in (None, "NULL") else scale_column
    top_scaled = scale_data(data, by=by)
    full_scaled = scale_data(log_data, by=by)
    return Features(cell_type, log_data, data, top_scaled, full_scaled)


# ----------------------------------------------------------------------------
# steps: NMF grid, combine restarts, rank selection
# ----------------------------------------------------------------------------
def nmf_input(top_scaled: pd.DataFrame, r_exact=True) -> np.ndarray:
    """``posneg(as.matrix(read.delim("expression_top_genes_scaled.txt")))``."""
    return posneg(_rt(top_scaled.to_numpy(dtype=np.float64), r_exact))


def run_nmf(V: np.ndarray, rank: int, restart: int) -> NMFFit:
    """``state_discovery_NMF.R``: ``nmf(V, rank, "brunet", seed = 1234 + restart)``."""
    return nmf_brunet(V, rank, 1234 + restart)


def rank_metrics(fits: list[NMFFit], rank: int):
    """``state_discovery_combine_NMF_restarts.R`` for one rank → (merged, metrics dict)."""
    merged = merge_runs(fits)
    sil = silhouettes(merged)
    return merged, {
        "Cophenetic": cophcor(merged.consensus),
        "Silhouette_s": sil["samples"],
        "Silhouette_g": sil["features"],
        "Silhouette_chc": sil["chc"],
        "Silhouette_con": sil["consensus"],
        "Dispersion": dispersion(merged.consensus),
        "n_clusters": rank,
    }


def select_ranks(rank_tables: dict, max_n_clusters: int, min_cophenetic: float, r_exact=True) -> pd.DataFrame:
    """``state_discovery_rank_selection.R``. ``rank_tables``: cell type → DataFrame
    with one row per rank (columns as rank_data.txt)."""
    chosen = {}
    drops = {}
    for ct in factor_levels(list(rank_tables)):
        spl = rank_tables[ct]
        coph = _rt(spl["Cophenetic"].to_numpy(dtype=np.float64), r_exact)
        ncl = spl["n_clusters"].to_numpy()
        n = len(coph)
        cross = [False, False] + [bool(coph[i] < min_cophenetic and coph[i - 1] >= min_cophenetic
                                       and coph[i - 2] >= min_cophenetic) for i in range(2, n)]
        wc = [i for i, c in enumerate(cross) if c]
        if not wc:
            idx = max_n_clusters  # 1-based row index, as in R (may be out of range -> NA)
        else:
            lc = wc[-1] + 1  # 1-based
            a = abs(min_cophenetic - coph[lc - 2])
            b = abs(min_cophenetic - coph[lc - 1])
            idx = lc - 2 + (1 if not (b < a) else 2)
        chosen[ct] = int(ncl[idx - 1]) if 1 <= idx <= n else None
        diffs = [np.nan] + [coph[i - 1] - coph[i] for i in range(1, n)]
        k = which_max(np.array(diffs))
        drops[ct] = int(ncl[k - 1]) if k >= 1 else None
    out = pd.DataFrame({"CellType": list(chosen), "Chosen_Rank": list(chosen.values())})
    redefined = [v is None for v in chosen.values()]
    out["Chosen_Rank"] = [drops[c] if v is None else v for c, v in chosen.items()]
    out["Redefined"] = redefined
    return out


# ----------------------------------------------------------------------------
# step: initial states (state_discovery_initial_plots.R)
# ----------------------------------------------------------------------------
def get_fc_for_clusters(data: pd.DataFrame, variable) -> pd.DataFrame:
    """``get_FC_for_clusters`` (lib/misc.R)."""
    variable = np.array([str(v) for v in variable], dtype=object)
    levels = factor_levels(list(variable))
    X = data.to_numpy(dtype=np.float64)
    fc = np.empty((X.shape[0], len(levels)))
    for k, lv in enumerate(levels):
        inc = variable == lv
        fc[:, k] = apply_mean_rows(X[:, inc], na_rm=True) - apply_mean_rows(X[:, ~inc], na_rm=True)
    st = []
    mx = []
    for i in range(fc.shape[0]):
        j = which_max(fc[i])
        if j < 0:
            st.append(None)
            mx.append(np.nan)
        else:
            st.append(levels[j])
            mx.append(fc[i, j])
    genes = [str(g) for g in data.index]
    df = pd.DataFrame({"Gene": genes}, index=pd.Index(genes, dtype=object))
    for k, lv in enumerate(levels):
        df[lv] = fc[:, k]
    df["State"] = pd.Series(st, index=df.index, dtype=object)
    df["MaxFC"] = mx
    o = r_order(df["State"].tolist(), (-df["MaxFC"]).tolist())
    return df.iloc[o]


@dataclass
class InitialStates:
    cell_type: str
    rank: int
    W: pd.DataFrame
    H: pd.DataFrame
    initial_abundances: pd.DataFrame
    initial_assignment: pd.DataFrame
    initial_gene_info: pd.DataFrame


def initial_states(merged, top_scaled: pd.DataFrame, cell_type: str, rank: int, r_exact=True,
                   gene_info: bool = True) -> InitialStates:
    """``state_discovery_initial_plots.R`` (``gene_info=False``: the scRNA
    cross-correlation variant, which does not compute marker genes)."""
    fit = merged.fit
    genes = [str(g) for g in top_scaled.index]
    samples = [str(s) for s in top_scaled.columns]
    states = [f"IS{k + 1:02d}" for k in range(fit.W.shape[1])]
    W = pd.DataFrame(fit.W, index=[g + "__pos" for g in genes] + [g + "__neg" for g in genes], columns=states)
    H = pd.DataFrame(fit.H, index=states, columns=samples)
    ab = pd.DataFrame(col_normalize(fit.H), index=states, columns=samples)
    # predict(): factor of which.max -> sprintf("IS%02d", <factor codes>)
    pred = predict_columns(fit.H)
    lv = np.unique(pred)
    codes = np.searchsorted(lv, pred) + 1
    cls = pd.DataFrame({"ID": samples, "State": [f"IS{c:02d}" for c in codes]})
    cls = cls.iloc[r_order(cls["State"].tolist())]
    gi = None
    if gene_info:
        raw = _rt_df(top_scaled, r_exact)
        data = raw.iloc[:, r_match(cls["ID"].tolist(), samples)]
        gi = get_fc_for_clusters(data, cls["State"].tolist())
    return InitialStates(cell_type, rank, W, H, ab, cls, gi)


# ----------------------------------------------------------------------------
# step: QC filter (state_discovery_first_filter.R)
# ----------------------------------------------------------------------------
@dataclass
class States:
    cell_type: str
    rank: int
    first_filter_data: pd.DataFrame
    mapping: pd.DataFrame
    state_abundances: pd.DataFrame
    state_assignment: pd.DataFrame
    heatmap_top_ann: pd.DataFrame
    gene_info: pd.DataFrame
    heatmap_data: pd.DataFrame
    log: list = field(default_factory=list)


def _mode_count(row: np.ndarray) -> int:
    vals = [format_real(float(v)) for v in row if v == v]
    if not vals:
        return 0
    from collections import Counter

    return max(Counter(vals).values())


def first_filter(init: InitialStates, feats: Features, annotation: pd.DataFrame | None, r_exact=True,
                 csx_filter: bool = True) -> States:
    """``state_discovery_first_filter.R`` (``csx_filter=False``:
    ``state_discovery_first_filter_scRNA.R``, without the filter on genes
    whose imputed values are mostly identical)."""
    raw = _rt_df(feats.top_scaled, r_exact)
    gi = init.initial_gene_info.copy()
    if csx_filter:
        csx = _rt_df(feats.top_log2, r_exact)
        gidx = r_match([str(g) for g in raw.index], [str(g) for g in csx.index])
        C = csx.to_numpy()
        ncol = C.shape[1]
        keep_genes = set()
        for gi_, g in zip(gidx, raw.index):
            row = C[gi_] if gi_ >= 0 else np.full(ncol, np.nan)
            if not (_mode_count(row) > 0.5 * ncol):
                keep_genes.add(str(g))
    else:
        keep_genes = {str(i) for i in gi.index}
    num_cols = [c for c in gi.columns if c not in ("Gene", "State")]
    for c in num_cols:
        gi[c] = _rt(gi[c].to_numpy(dtype=np.float64), r_exact)
    gi = gi[[str(i) in keep_genes for i in gi.index]]
    classes = init.initial_assignment.copy()
    classes.index = pd.Index([str(i + 1) for i in range(len(classes))], dtype=object)
    Wv = _rt(init.W.to_numpy(), r_exact)
    rn = list(init.W.index)
    neg = np.array(["_neg" in r for r in rn])
    pos = np.array(["_pos" in r for r in rn])
    agg = {s: r_sum(Wv[neg, k]) / r_sum(Wv[pos, k]) for k, s in enumerate(init.W.columns)}
    st_levels = factor_levels([s for s in gi["State"].tolist() if s is not None and s == s])
    counts = {s: int((gi["State"] == s).sum()) for s in st_levels}
    tb = pd.DataFrame({"State": st_levels, "NumberMarkers": [counts[s] for s in st_levels]})
    ag = pd.DataFrame({"InitialState": list(agg), "PosRatio": list(agg.values())})
    tb = r_merge(tb, ag.rename(columns={"InitialState": "State"}), "State", all_y=True,
                  x_factor=True)
    tb["NumberMarkers"] = [0 if pd.isna(v) else int(v) for v in tb["NumberMarkers"]]
    log = []
    to_keep = [str(s) for s, n in zip(tb["State"], tb["NumberMarkers"]) if n > 10]
    if len(to_keep) < 2:
        log.append("fewer than two states with > 10 markers; marker filter not applied")
        to_keep = [str(s) for s in tb["State"]]
    to_keep2 = [str(s) for s, r in zip(tb["State"], tb["PosRatio"]) if r < 1]
    inter = [s for s in dict.fromkeys(to_keep2) if s in set(to_keep)]
    if len(inter) < 2:
        log.append("fewer than two states pass the AFI filter; AFI filter not applied")
    else:
        to_keep = inter
    ks = set(to_keep)
    classes = classes[[s in ks for s in classes["State"]]]
    gi = gi[[s in ks for s in gi["State"]]]
    samples = [str(s) for s in raw.columns]
    data = raw.iloc[:, r_match(classes["ID"].tolist(), samples)]
    data = data.iloc[r_match(gi["Gene"].tolist(), [str(g) for g in data.index])]
    lv = factor_levels(classes["State"].tolist())
    X = data.to_numpy()
    st_arr = np.array(classes["State"].tolist(), dtype=object)
    agg_data = np.column_stack([apply_mean_rows(X[:, st_arr == l], na_rm=True) for l in lv])
    h = hclust(dist_euclidean_rows(agg_data.T), len(lv), "complete")
    init_order = [lv[i - 1] for i in h.order]
    mapping = {s: f"S{k + 1:02d}" for k, s in enumerate(init_order)}
    mp = pd.DataFrame({"State": [mapping[s] for s in init_order], "InitialState": init_order})
    H0 = _rt_df(init.initial_abundances, r_exact)
    sa = H0.iloc[r_match(init_order, [str(i) for i in H0.index])]
    sa.index = pd.Index(mp["State"].tolist(), dtype=object)
    classes = classes.copy()
    classes["InitialState"] = classes["State"]
    classes["State"] = [mapping[s] for s in classes["State"]]
    classes = classes.iloc[r_order(classes["State"].tolist())]
    data = raw.iloc[:, r_match(classes["ID"].tolist(), samples)]
    clin = read_clinical([str(c) for c in data.columns], annotation)
    top = r_merge(clin.reset_index(drop=True), classes.reset_index(drop=True), "ID", all_y=True)
    top.index = pd.Index(top["ID"].tolist(), dtype=object)
    top = top.iloc[r_match([str(c) for c in data.columns], top["ID"].tolist())]
    gi = gi.copy()
    gi["InitialState"] = gi["State"]
    gi["State"] = [mapping[s] for s in gi["State"]]
    gi = gi.iloc[r_order(gi["State"].tolist())]
    data = data.iloc[r_match(gi["Gene"].tolist(), [str(g) for g in data.index])]
    return States(init.cell_type, init.rank, tb, mp, sa, classes, top, gi, data, log)


# ----------------------------------------------------------------------------
# ecotypes (ecotypes.R)
# ----------------------------------------------------------------------------
@dataclass
class Ecotypes:
    mapping_all_states: pd.DataFrame
    binary: pd.DataFrame
    jaccard: pd.DataFrame
    silhouette_initial: float
    initial_ecotypes: pd.DataFrame
    ecotypes: pd.DataFrame
    silhouette: float
    n_clusters: int


def discover_ecotypes(initial: dict, states: dict, cell_types: list[str], min_states: int = 3) -> Ecotypes:
    """``ecotypes.R``. ``initial``/``states``: cell type → InitialStates/States."""
    maps = []
    all_classes = None
    for ct in cell_types:
        m = states[ct].mapping[["State", "InitialState"]].copy()
        m["CellType"] = ct
        maps.append(m)
        cl = initial[ct].initial_assignment[["ID", "State"]].copy()
        cl.columns = ["ID", ct]
        all_classes = cl if all_classes is None else r_merge(all_classes, cl.reset_index(drop=True), "ID")
    am = pd.concat(maps, ignore_index=True)
    am["InitialID"] = am["CellType"] + "_" + am["InitialState"]
    am["ID"] = am["CellType"] + "_" + am["State"]
    ids = all_classes["ID"].tolist()
    rows = []
    rnames = []
    for ct in cell_types:
        v = np.array([str(x) for x in all_classes[ct]], dtype=object)
        for val in factor_levels(list(v)):
            rows.append(v == val)
            rnames.append(f"{ct}_{val}")
    B = np.array(rows)
    sel = r_match(am["InitialID"].tolist(), rnames)
    clusters = B[sel]
    N = len(ids)
    n = clusters.shape[0]
    J = np.full((n, n), np.nan)
    sums = clusters.sum(axis=1)
    for i in range(n):
        for j in range(i, n):
            inter = int((clusters[i] & clusters[j]).sum())
            idx = inter / (sums[i] + sums[j] - inter)
            p = phyper_upper(inter, sums[i], N - sums[i], sums[j])
            if p > 0.01:
                idx = 0.0
            J[i, j] = J[j, i] = idx
    sid = am["ID"].tolist()
    jac = pd.DataFrame(J, index=sid, columns=sid)
    binary = pd.DataFrame(clusters, index=sid, columns=ids)
    D = 1.0 - J
    d = np.concatenate([D[j + 1:, j] for j in range(n - 1)])
    hc = hclust(d, n, "average")
    best_k, best = None, -np.inf
    sils = []
    for k in range(2, n):
        s = silhouette_avg_width_dist(cutree(hc, k), d)
        sils.append(s)
    k_opt = 2 + which_max(np.array(sils))
    clust = cutree(hc, k_opt)
    sil0 = silhouette_avg_width_dist(clust, d)
    parts = [s.rsplit("_", 1) for s in sid]
    top = pd.DataFrame({"CellType": [p[0] for p in parts], "State": [p[1] for p in parts],
                        "InitialEcotype": [f"IE{c:02d}" for c in clust]}, index=pd.Index(sid, dtype=object))
    init_eco = top.copy()
    tb = {lv: int((top["InitialEcotype"] == lv).sum()) for lv in factor_levels(top["InitialEcotype"].tolist())}
    top = top[[tb[x] >= min_states for x in top["InitialEcotype"]]].copy()
    top["ID"] = list(top.index)
    nm = list(dict.fromkeys(top["InitialEcotype"].tolist()))
    emap = {x: f"E{k + 1}" for k, x in enumerate(nm)}
    top["Ecotype"] = [emap[x] for x in top["InitialEcotype"]]
    lvls = ecotype_levels(top["Ecotype"].tolist())
    code = r_match(top["Ecotype"].tolist(), lvls)
    top = top.iloc[np.argsort(code, kind="stable")]
    sel = r_match(top["ID"].tolist(), sid)
    D2 = D[np.ix_(sel, sel)]
    d2 = np.concatenate([D2[j + 1:, j] for j in range(len(sel) - 1)]) if len(sel) > 1 else np.empty(0)
    cl2 = [int(e.replace("E", "")) for e in top["Ecotype"]]
    sil = silhouette_avg_width_dist(np.array(cl2), d2) if len(sel) > 1 else np.nan
    return Ecotypes(am[["State", "InitialState", "CellType", "InitialID", "ID"]], binary, jac, sil0, init_eco,
                    top.reset_index(drop=True), sil, int(k_opt))


def assign_samples(states: dict, eco: pd.DataFrame, cell_types: list[str], annotation: pd.DataFrame | None,
                   r_exact=True):
    """``ecotypes_assign_samples.R`` (discovery cohort)."""
    from .recovery_steps import EcotypeRecovery, _ecotype_means, _sorted_ecotypes, _ttest_matrix

    eco_sorted, levels = _sorted_ecotypes(eco)
    blocks, rown = [], []
    assigned: set = set()
    samples = None
    for ct in cell_types:
        s = states[ct]
        ab = s.state_abundances
        if samples is None:
            samples = [str(c) for c in ab.columns]
        else:
            ab = ab[samples]
        blocks.append(_rt(ab.to_numpy(), r_exact))
        rown += [f"{ct}_{x}" for x in ab.index]
        st_in = set(eco_sorted.loc[eco_sorted["CellType"] == ct, "State"].astype(str))
        assigned |= {i for i, x in zip(s.state_assignment["ID"], s.state_assignment["State"]) if x in st_in}
    allH = np.vstack(blocks)
    idx = r_match([str(i) for i in eco_sorted["ID"]], rown)
    all_H = np.full((len(idx), allH.shape[1]), np.nan)
    all_H[idx >= 0] = allH[idx[idx >= 0]]
    row_ids = [rown[k] if k >= 0 else "NA" for k in idx]
    H = col_normalize(_ecotype_means(all_H, row_ids, eco_sorted, levels, na_rm=False))
    p = _ttest_matrix(all_H, row_ids, eco_sorted, levels)
    best = [which_max(H[:, j]) for j in range(H.shape[1])]
    clinical = pd.DataFrame({"ID": samples, "MaxEcotype": [levels[b] for b in best],
                             "AssignmentP": [p[b, j] for j, b in enumerate(best)]})
    clinical["AssignmentQ"] = p_adjust_bh(clinical["AssignmentP"].to_numpy())
    clinical["AssignedToEcotypeStates"] = [s in assigned for s in samples]
    clinical["Ecotype"] = [m if (q < 0.25 and a) else "Unassigned" for m, q, a in
                           zip(clinical["MaxEcotype"], clinical["AssignmentQ"], clinical["AssignedToEcotypeStates"])]
    tmp = read_clinical(samples, annotation)
    to_rem = [c for c in tmp.columns if c in clinical.columns and c != "ID"]
    tmp = tmp[[c for c in tmp.columns if c not in to_rem]]
    clinical = r_merge(clinical, tmp.reset_index(drop=True), "ID", all_x=True)
    clinical.index = pd.Index([str(i + 1) for i in range(len(clinical))], dtype=object)
    clinical = clinical.iloc[r_order(clinical["MaxEcotype"].tolist())]
    filt = clinical[clinical["Ecotype"] != "Unassigned"].copy()
    filt.index = pd.Index(filt["ID"].tolist(), dtype=object)
    return EcotypeRecovery(combined_state_abundances=pd.DataFrame(all_H, index=row_ids, columns=samples),
                           ecotype_abundance=pd.DataFrame(H, index=levels, columns=samples),
                           assignment_p_vals=pd.DataFrame(p, index=levels, columns=samples),
                           initial_assignment=clinical, assignment=filt, ecotypes=eco_sorted)
