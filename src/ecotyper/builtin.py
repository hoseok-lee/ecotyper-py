"""The prebuilt EcoTyper models (Carcinoma, Lymphoma) by name.

The models live in the R repository (``digitalcytometry/ecotyper``, folder
``EcoTyper/<name>/<name>_Fractions``). Recovery only needs a few small files
per cell type (about 5 MB per model instead of ~440 MB for the whole folder),
so these are downloaded on first use from a pinned commit and cached.

Lookup order for :func:`builtin_model_path`:

1. ``$ECOTYPER_MODELS``: a local ``EcoTyper`` folder of the R repository
   (or any folder containing ``<name>/<name>_Fractions``);
2. the cache ``$ECOTYPER_CACHE`` (default ``~/.cache/ecotyper``);
3. download into the cache.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

#: EcoTyper R repository commit the models are taken from (the one validated against).
COMMIT = "11408e71900e15829cf4f33e1331ce47c053472d"
REPO_RAW = "https://raw.githubusercontent.com/digitalcytometry/ecotyper"
BUILTIN = ("Carcinoma", "Lymphoma")

#: files of ``<fractions folder>`` that recovery reads (heatmap_data.txt is not
#: needed: its row names equal the genes of gene_info.txt in every shipped model)
MODEL_FILES = ("Analysis/rank_selection/rank_data.txt", "Ecotypes/discovery/ecotypes.txt",
               "Ecotypes/discovery/mapping_all_states.txt")
STATE_FILES = ("W.txt", "mapping_to_initial_states.txt", "initial_gene_info.txt", "gene_info.txt")


def cache_dir() -> Path:
    return Path(os.environ.get("ECOTYPER_CACHE", Path.home() / ".cache" / "ecotyper"))


def _local(name: str) -> Path | None:
    root = os.environ.get("ECOTYPER_MODELS")
    if not root:
        return None
    for cand in (Path(root) / name / f"{name}_Fractions", Path(root) / "EcoTyper" / name / f"{name}_Fractions"):
        if (cand / "Analysis").is_dir():
            return cand
    return None


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def builtin_model_path(name: str, *, download: bool = True, verbose: bool = True) -> Path:
    """Folder ``.../<name>/<name>_Fractions`` of a prebuilt model, downloading it if needed."""
    if name not in BUILTIN:
        raise ValueError(f"Unknown built-in model {name!r}; available: {', '.join(BUILTIN)}")
    local = _local(name)
    if local is not None:
        return local
    base = cache_dir() / "models" / COMMIT[:12] / name / f"{name}_Fractions"
    done = base / ".complete"
    if done.exists():
        return base
    if not download:
        raise FileNotFoundError(f"Built-in model {name!r} is not cached in {base}")
    url = f"{REPO_RAW}/{COMMIT}/EcoTyper/{name}/{name}_Fractions"
    if verbose:
        print(f"Downloading the {name} model from github.com/digitalcytometry/ecotyper into {base}")
    for f in MODEL_FILES:
        if not (base / f).exists():
            _fetch(f"{url}/{f}", base / f)
    from .rcompat.io import read_delim

    rank = read_delim(base / MODEL_FILES[0])
    for ct, n in zip(rank.iloc[:, 0], rank.iloc[:, 1]):
        rel = f"Cell_States/discovery/{ct}/{int(n)}"
        for f in STATE_FILES:
            if not (base / rel / f).exists():
                _fetch(f"{url}/{rel}/{f}", base / rel / f)
    done.write_text(COMMIT + "\n")
    return base
