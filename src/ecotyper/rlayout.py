"""Write results in the R pipeline's directory layout and file formats
(``write.table(x, sep = "\\t")``), so outputs can be diffed with R's."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .rcompat.io import write_matrix, write_table
from .recovery_steps import EcotypeRecovery, StateRecovery


def _wm(df: pd.DataFrame, path) -> None:
    write_matrix(df.to_numpy(), [str(i) for i in df.index], [str(c) for c in df.columns], path)


def write_state_recovery(res: StateRecovery, out_dir) -> None:
    """Files of ``Cell_States/recovery/<dataset>/<cell type>/<rank>/``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    _wm(res.expression, d / "expression_matrix.txt")
    if res.annotation is not None:
        write_table(res.annotation, d / "annotation.txt")
    _wm(res.scaled, d / "expression_matrix_scaled.txt")
    _wm(res.H, d / "H.txt")
    _wm(res.initial_abundances, d / "initial_state_abundances.txt")
    write_table(res.initial_assignment, d / "initial_state_assignment.txt", row_names=False)
    _wm(res.abundances, d / "state_abundances.txt")
    write_table(res.assignment, d / "state_assignment.txt", row_names=False)
    if res.heatmap_data is not None:
        _wm(res.heatmap_data, d / "heatmap_data.txt")
    if res.heatmap_top_ann is not None:
        write_table(res.heatmap_top_ann, d / "heatmap_top_ann.txt")


def write_ecotype_recovery(res: EcotypeRecovery, out_dir) -> None:
    """Files of ``Ecotypes/recovery/<dataset>/``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    _wm(res.combined_state_abundances, d / "combined_state_abundances.txt")
    _wm(res.ecotype_abundance, d / "ecotype_abundance.txt")
    _wm(res.assignment_p_vals, d / "assignment_p_vals.txt")
    write_table(res.initial_assignment, d / "initial_ecotype_assignment.txt")
    write_table(res.assignment, d / "ecotype_assignment.txt")


def write_recovery_output(state_results: dict, eco: EcotypeRecovery | None, out_dir, cell_types,
                          state_files=("state_abundances.txt", "state_assignment.txt", "heatmap_data.txt",
                                       "heatmap_top_ann.txt")) -> None:
    """The user-facing ``RecoveryOutput/<dataset>/`` folder (subset of files copied by the R wrappers)."""
    d = Path(out_dir)
    for ct in cell_types:
        if ct not in state_results:
            continue
        r = state_results[ct]
        cd = d / ct
        cd.mkdir(parents=True, exist_ok=True)
        if "state_abundances.txt" in state_files:
            _wm(r.abundances, cd / "state_abundances.txt")
        if "state_assignment.txt" in state_files:
            write_table(r.assignment, cd / "state_assignment.txt", row_names=False)
        if "heatmap_data.txt" in state_files and r.heatmap_data is not None:
            _wm(r.heatmap_data, cd / "heatmap_data.txt")
        if "heatmap_top_ann.txt" in state_files and r.heatmap_top_ann is not None:
            write_table(r.heatmap_top_ann, cd / "heatmap_top_ann.txt")
    if eco is not None:
        ed = d / "Ecotypes"
        ed.mkdir(parents=True, exist_ok=True)
        write_table(eco.assignment, ed / "ecotype_assignment.txt")
        _wm(eco.ecotype_abundance, ed / "ecotype_abundance.txt")
