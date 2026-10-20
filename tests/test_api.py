"""The scverse-style API (:func:`ecotyper.recovery`, :func:`ecotyper.discovery`,
:class:`ecotyper.EcoTyperModel`) gives results identical to the file-based
workflows it wraps."""
import gzip
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import ecotyper as et
from conftest import R_WORK
from ecotyper import workflows as wf
from ecotyper.io import read_annotation, read_bulk_matrix, read_fractions, read_scrna_matrix, to_anndata

EX = R_WORK / "example_data"
MODELS = R_WORK / "EcoTyper"


@pytest.fixture(autouse=True)
def _local_models(monkeypatch):
    if MODELS.exists():
        monkeypatch.setenv("ECOTYPER_MODELS", str(MODELS))


def need_examples():
    if not (EX / "scRNA_lymphoma_data.txt").exists() or not (MODELS / "Lymphoma").exists():
        pytest.skip("EcoTyper example data / models not available (R working copy)")


def same(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def check_states(adata, states, key="ecotyper", id_map=None):
    for ct, r in states.items():
        ids = [str(c) for c in r.abundances.columns]
        if id_map is not None:
            ids = [id_map[i] for i in ids]
        got = adata.obsm[f"{key}_abundance_{ct}"].loc[ids]
        assert same(got.to_numpy(), r.abundances.T.to_numpy()), ct
        a = r.assignment
        aids = [str(i) for i in a["ID"]] if id_map is None else [id_map[str(i)] for i in a["ID"]]
        assert list(adata.obs.loc[aids, f"{key}_state_{ct}"].astype(str)) == [str(s) for s in a["State"]], ct


# ----------------------------------------------------------------------------
# data type inference
# ----------------------------------------------------------------------------
def _toy(n_obs=20, n_var=30, sparse=False, **obs):
    X = np.random.default_rng(0).poisson(1.0, (n_obs, n_var)).astype(float)
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    for k, v in obs.items():
        a.obs[k] = v
    return a


def test_infer_data_type():
    assert et.infer_data_type(_toy()) == "bulk"
    assert et.infer_data_type(_toy(CellType=["B"] * 20)) == "scRNA"
    assert et.infer_data_type(_toy(cell_type=["B"] * 20), cell_type_key="cell_type") == "scRNA"
    v = _toy()
    v.obsm["spatial"] = np.zeros((20, 2))
    assert et.infer_data_type(v) == "visium"
    assert et.infer_data_type({"B.cells": _toy()}) == "profiles"
    v.obs["CellType"] = "B"
    with pytest.raises(ValueError, match="Cannot tell"):
        et.infer_data_type(v)
    with pytest.raises(ValueError, match="single-cell"):
        et.infer_data_type(_toy(n_obs=2500, sparse=True))


def test_discovery_rejects_bulk_and_visium():
    with pytest.raises(ValueError, match="CIBERSORTx"):
        et.discovery(_toy())
    v = _toy()
    v.obsm["spatial"] = np.zeros((20, 2))
    with pytest.raises(ValueError, match="Visium"):
        et.discovery(v)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def test_builtin_model_save_load(tmp_path):
    need_examples()
    m = et.EcoTyperModel.load("Lymphoma")
    assert m.name == "Lymphoma" and len(m.cell_types) == 13
    out = m.save(tmp_path / "Lymphoma" / "Lymphoma_Fractions", heatmap=False)
    m2 = et.EcoTyperModel.load(out)
    assert m2.cell_type_order == m.cell_type_order
    for ct in m.cell_type_order:
        a, b = m.cell_types[ct], m2.cell_types[ct]
        assert same(a.W.to_numpy(), b.W.to_numpy()) and list(a.W.index) == list(b.W.index)
        assert a.mapping.equals(b.mapping)
        assert a.heatmap_genes == b.heatmap_genes  # from gene_info.txt when heatmap_data.txt is absent
    assert m.ecotypes.equals(m2.ecotypes)


def test_builtin_download(tmp_path, monkeypatch):
    if os.environ.get("ECOTYPER_TEST_DOWNLOAD") != "1":
        pytest.skip("set ECOTYPER_TEST_DOWNLOAD=1 to test downloading a built-in model")
    need_examples()
    monkeypatch.delenv("ECOTYPER_MODELS", raising=False)
    monkeypatch.setenv("ECOTYPER_CACHE", str(tmp_path))
    m = et.EcoTyperModel.load("Lymphoma")
    ref = et.EcoTyperModel.load(MODELS / "Lymphoma" / "Lymphoma_Fractions")
    for ct in ref.cell_type_order:
        assert same(m.cell_types[ct].W.to_numpy(), ref.cell_types[ct].W.to_numpy())
        assert m.cell_types[ct].heatmap_genes == ref.cell_types[ct].heatmap_genes


# ----------------------------------------------------------------------------
# recovery == file-based workflows
# ----------------------------------------------------------------------------
def test_recovery_scrna_sparse_matches_workflow():
    need_examples()
    model = et.EcoTyperModel.load("Lymphoma")
    expr = read_scrna_matrix(EX / "scRNA_lymphoma_data.txt")
    ann = read_annotation(EX / "scRNA_lymphoma_annotation.txt")
    states, _ = wf.recover_scrna_frame(expr, ann, model)
    a = to_anndata(expr, ann)
    a.X = sp.csr_matrix(a.X)
    a.obs = a.obs.rename(columns={"CellType": "cell_type"})
    et.recovery(a, "Lymphoma", cell_type_key="cell_type")
    assert a.uns["ecotyper"]["params"]["data_type"] == "scRNA"
    check_states(a, states)
    assert a.obs["ecotyper_ecotype"].notna().any()


def test_recovery_scrna_ecotypes_per_sample(tmp_path):
    need_examples()
    if not (EX / "scRNA_CRC_data.txt").exists():
        pytest.skip("scRNA_CRC_data.txt not extracted")
    model = et.EcoTyperModel.load("Carcinoma")
    expr = read_scrna_matrix(EX / "scRNA_CRC_data.txt")
    ann = read_annotation(EX / "scRNA_CRC_annotation.txt")
    states, eco = wf.recover_scrna_frame(expr, ann, model, n_jobs=4)
    a = to_anndata(expr, ann)
    out = et.recovery(a, model, data_type="scRNA", copy=True, n_jobs=4)
    assert "ecotyper" not in a.uns
    check_states(out, states)
    assert same(out.uns["ecotyper"]["ecotype_abundance"].to_numpy(), eco.ecotype_abundance.T.to_numpy())
    out.write_h5ad(tmp_path / "crc.h5ad")
    back = ad.read_h5ad(tmp_path / "crc.h5ad")
    assert same(back.uns["ecotyper"]["ecotype_abundance"].to_numpy(), eco.ecotype_abundance.T.to_numpy())


def test_recovery_bulk_matches_workflow():
    need_examples()
    model = et.EcoTyperModel.load("Lymphoma")
    expr = read_bulk_matrix(EX / "bulk_lymphoma_data.txt")
    ann = read_annotation(EX / "bulk_lymphoma_annotation.txt")
    states, eco = wf.recover_bulk_frame(expr, model, ann)
    a = to_anndata(expr, ann)
    assert et.infer_data_type(a) == "bulk"
    et.recovery(a, model, key_added="lym")
    check_states(a, states, key="lym")
    ia = eco.initial_assignment.set_index("ID")
    assert list(a.obs["lym_ecotype"].astype(str)) == [str(v) for v in ia["Ecotype"].reindex(a.obs_names)]
    assert same(a.obsm["lym_ecotype_abundance"].to_numpy(), eco.ecotype_abundance.T.reindex(a.obs_names).to_numpy())


def _visium_anndata():
    expr, spots = wf.read_visium(EX / "VisiumBreast")
    with gzip.open(EX / "VisiumBreast" / "barcodes.tsv.gz", "rt") as fh:
        bcs = [line.split("\t")[0].strip() for line in fh if line.strip()]
    a = ad.AnnData(sp.csr_matrix(expr.to_numpy().T), obs=pd.DataFrame(index=bcs),
                   var=pd.DataFrame(index=[str(g) for g in expr.index]))
    a.obs["array_row"] = spots["X"].to_numpy().astype(int)
    a.obs["array_col"] = spots["Y"].to_numpy().astype(int)
    a.obsm["spatial"] = np.c_[spots["Y"].to_numpy(), spots["X"].to_numpy()]
    a.uns["spatial"] = {"VisiumBreast": {}}
    return a, expr, spots


def test_recovery_visium_matches_workflow(tmp_path):
    need_examples()
    from ecotyper.rcompat.frame import make_names

    model = et.EcoTyperModel.load("Carcinoma")
    a, expr, spots = _visium_anndata()
    frac = read_fractions(EX / "visium_fractions_example.txt")
    states, long, wide, eco = wf.recover_visium_frame(expr, spots, model, frac, sample_name="VisiumBreast", n_jobs=4)
    # user-facing fractions: indexed by the AnnData spot names (barcodes with "-1")
    ids = make_names(list(a.obs_names))
    to_obs = dict(zip(ids, a.obs_names))
    f = frac.set_index(frac.columns[0])
    f.index = [to_obs[i] for i in f.index]
    a.obsm["csx"] = f.reindex(a.obs_names)
    et.recovery(a, model, fractions="csx", n_jobs=4)
    assert a.uns["ecotyper"]["params"]["data_type"] == "visium"
    check_states(a, states, id_map=to_obs)
    w = wide.set_index(wide["ID"].map(to_obs))
    cols = [c for c in wide.columns if c not in ("ID", "X", "Y", "Sample", "Malignant")]
    assert same(a.obsm["ecotyper_state_abundance"][cols].to_numpy(), w.loc[a.obs_names, cols].to_numpy())
    e = eco.set_index(eco["ID"].map(to_obs))
    ecols = [c for c in eco.columns if c not in ("ID", "X", "Y", "Sample", "Malignant")]
    assert same(a.obsm["ecotyper_ecotype_abundance"][ecols].to_numpy(), e.loc[a.obs_names, ecols].to_numpy())
    a.write_h5ad(tmp_path / "visium.h5ad")


# ----------------------------------------------------------------------------
# discovery == file-based workflows (small configurations)
# ----------------------------------------------------------------------------
PROFILE_CTS = ["B.cells", "CD8.T.cells", "Fibroblasts", "Monocytes.and.Macrophages"]


def test_discovery_profiles_model_matches_files(tmp_path):
    need_examples()
    d = EX / "Tutorial_6" / "PresortedDiscovery"
    profiles = wf.read_profiles(d, PROFILE_CTS)
    ann = read_annotation(EX / "Tutorial_6" / "PresortedDiscovery_annotation.txt")
    kw = dict(max_states=4, restarts=2, n_jobs=4)
    res = wf.discover_from_profiles(profiles, ann, **kw)
    wf.write_discovery(res, tmp_path / "files", "small", "All_genes")
    ref = et.EcoTyperModel.load(tmp_path / "files" / "EcoTyper" / "small" / "All_genes")

    adatas = {ct: to_anndata(p, ann) for ct, p in profiles.items()}
    model = et.discovery(adatas, annotation=ann, name="small", **kw)
    assert model.cell_type_order == ref.cell_type_order
    for ct in ref.cell_type_order:
        assert same(model.cell_types[ct].W.to_numpy(), ref.cell_types[ct].W.to_numpy()), ct
        assert model.cell_types[ct].mapping.equals(ref.cell_types[ct].mapping)
        assert model.cell_types[ct].heatmap_genes == ref.cell_types[ct].heatmap_genes
    assert model.ecotypes.equals(ref.ecotypes)
    # save() writes the same bytes as the file-based workflow
    out = model.save(tmp_path / "saved")
    base = tmp_path / "files" / "EcoTyper" / "small" / "All_genes"
    for f in out.rglob("*.txt"):
        assert f.read_bytes() == (base / f.relative_to(out)).read_bytes(), f
    # and recovery with the in-memory model equals recovery with the file-loaded model
    expr = read_bulk_matrix(EX / "bulk_lung_data.txt")
    s1, e1 = wf.recover_bulk_frame(expr, ref, None)
    a = to_anndata(expr)
    et.recovery(a, model)
    check_states(a, s1)


def test_discovery_scrna_matches_workflow():
    need_examples()
    if not (EX / "scRNA_CRC_data.txt").exists():
        pytest.skip("scRNA_CRC_data.txt not extracted")
    expr = read_scrna_matrix(EX / "scRNA_CRC_data.txt")
    ann = read_annotation(EX / "scRNA_CRC_annotation.txt")
    keep = ann[ann["CellType"].isin(["B.cells", "Fibroblasts", "Endothelial.cells"])]
    keep = keep.groupby("CellType", group_keys=False).head(150).reset_index(drop=True)
    sub = expr[[str(i) for i in keep["ID"]]]
    kw = dict(max_states=3, restarts=2, n_jobs=4, verbose=False)
    res = wf.discover_scrna_frame(sub, keep, **kw)

    a = to_anndata(sub, keep)
    a.X = sp.csr_matrix(a.X)
    a.obs = a.obs.rename(columns={"CellType": "cell_type", "Sample": "patient"})
    model = et.discovery(a, cell_type_key="cell_type", sample_key="patient", **kw)
    r2 = model.discovery_result
    assert list(r2.ranks["Chosen_Rank"]) == list(res.ranks["Chosen_Rank"])
    for ct, st in res.states.items():
        assert same(r2.states[ct].state_abundances.to_numpy(), st.state_abundances.to_numpy()), ct
        sa = st.state_assignment
        assert list(a.obs.loc[sa["ID"].astype(str), f"ecotyper_state_{ct}"].astype(str)) == list(sa["State"]), ct
    assert r2.ecotypes.ecotypes.equals(res.ecotypes.ecotypes)
