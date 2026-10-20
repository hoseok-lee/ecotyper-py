"""``ecotyper-py`` command line, mirroring the R wrapper scripts."""
from __future__ import annotations

import argparse


def main(argv=None):
    from . import workflows as wf

    ap = argparse.ArgumentParser(prog="ecotyper-py", description="Python port of EcoTyper (R-exact).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-d", "--discovery", default="Carcinoma",
                       help="model: a built-in name (Carcinoma, Lymphoma), a model folder saved with "
                            "EcoTyperModel.save, or a discovery name under --models")
        p.add_argument("--models", default=None,
                       help="folder containing <discovery>/<discovery>_Fractions models (e.g. the 'EcoTyper' "
                            "folder of the R repository); default: built-in models, downloaded on first use")
        p.add_argument("-o", "--output", default="ecotyper_out")
        p.add_argument("-t", "--threads", type=int, default=1)
        p.add_argument("--seed", type=int, default=42, help="seed for NMFpredict's random initialisation")
        fast(p)

    def solver(p):
        p.add_argument("--nmf-method", choices=["brunet", "cd"], default="brunet",
                       help="NMF solver: brunet (EcoTyper/R) or cd (scikit-learn coordinate descent, "
                            "Frobenius loss; not R-equivalent)")

    def fast(p):
        p.add_argument("--fast", action="store_true",
                       help="fast mode (exact=False): BLAS NMF updates, not bit-identical to R")

    b = sub.add_parser("recover-bulk", help="EcoTyper_recovery_bulk.R")
    b.add_argument("-m", "--matrix", required=True)
    b.add_argument("-a", "--annotation", default=None)
    common(b)
    s = sub.add_parser("recover-scrna", help="EcoTyper_recovery_scRNA.R")
    s.add_argument("-m", "--matrix", required=True)
    s.add_argument("-a", "--annotation", required=True)
    s.add_argument("-s", "--subsample", type=int, default=-1)
    common(s)
    v = sub.add_parser("recover-visium", help="EcoTyper_recovery_visium.R (precomputed fractions)")
    v.add_argument("-i", "--input-dir", required=True)
    v.add_argument("-f", "--fractions", required=True)
    v.add_argument("-n", "--name", required=True)
    v.add_argument("--background", default="Epithelial.cells")
    common(v)
    d = sub.add_parser("discover", help="discovery from purified/presorted profiles (bulk downstream of CSx, presorted)")
    d.add_argument("-i", "--input-dir", required=True, help="folder with <cell type>.txt profiles")
    d.add_argument("-a", "--annotation", default=None)
    d.add_argument("-n", "--name", required=True)
    d.add_argument("--fractions", default="All_genes")
    d.add_argument("--scale-column", default=None)
    d.add_argument("--max-states", type=int, default=20)
    d.add_argument("--restarts", type=int, default=5)
    d.add_argument("--cophenetic", type=float, default=0.95)
    d.add_argument("--min-states", type=int, default=3)
    d.add_argument("-o", "--output", default="ecotyper_out")
    d.add_argument("-t", "--threads", type=int, default=1)
    fast(d)
    solver(d)
    r = sub.add_parser("discover-scrna", help="EcoTyper_discovery_scRNA.R")
    r.add_argument("-m", "--matrix", required=True)
    r.add_argument("-a", "--annotation", required=True)
    r.add_argument("-n", "--name", required=True)
    r.add_argument("--filter-genes", default="cell type specific")
    r.add_argument("--max-states", type=int, default=20)
    r.add_argument("--restarts", type=int, default=5)
    r.add_argument("--cophenetic", type=float, default=0.975)
    r.add_argument("--min-states", type=int, default=3)
    r.add_argument("--pvalue", type=float, default=1.0)
    r.add_argument("-o", "--output", default="ecotyper_out")
    r.add_argument("-t", "--threads", type=int, default=1)
    fast(r)
    solver(r)
    a = ap.parse_args(argv)
    exact = not a.fast

    if a.cmd == "recover-bulk":
        wf.recover_bulk_files(a.matrix, a.annotation, a.discovery, a.output, discovery_root=a.models,
                              seed=a.seed, n_jobs=a.threads, exact=exact)
    elif a.cmd == "recover-scrna":
        wf.recover_scrna_files(a.matrix, a.annotation, a.discovery, a.output, discovery_root=a.models,
                               seed=a.seed, subsample=a.subsample, n_jobs=a.threads, exact=exact)
    elif a.cmd == "recover-visium":
        wf.recover_visium_files(a.input_dir, a.fractions, a.discovery, a.output, recovery_name=a.name,
                                discovery_root=a.models, malignant_cell=a.background, seed=a.seed,
                                n_jobs=a.threads, exact=exact)
    elif a.cmd == "discover":
        wf.discover_files(a.input_dir, a.annotation, a.output, name=a.name, fractions=a.fractions,
                          scale_column=a.scale_column, max_states=a.max_states, restarts=a.restarts,
                          cophenetic_cutoff=a.cophenetic, min_states=a.min_states, n_jobs=a.threads,
                          exact=exact, nmf_method=a.nmf_method)
    elif a.cmd == "discover-scrna":
        from .io import read_annotation, read_scrna_matrix

        res = wf.discover_scrna_frame(read_scrna_matrix(a.matrix), read_annotation(a.annotation),
                                      filter_genes=a.filter_genes, max_states=a.max_states, restarts=a.restarts,
                                      cophenetic_cutoff=a.cophenetic, p_value_cutoff=a.pvalue,
                                      min_states=a.min_states, n_jobs=a.threads, exact=exact,
                                      nmf_method=a.nmf_method)
        frac = {"cell type specific": "Cell_type_specific_genes", "no filter": "All_genes"}.get(
            a.filter_genes, f"Top_{a.filter_genes}")
        wf.write_discovery_scrna(res, a.output, a.name, frac)


if __name__ == "__main__":
    main()
