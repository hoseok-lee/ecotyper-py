"""String collation as used by R's ``sort``/``order``/``factor``/``merge``.

In a non-C locale R (built with ICU, as the conda-forge and CRAN macOS builds
are) compares strings with an ICU collator for the current locale; in the C
locale it compares bytes. EcoTyper's output ordering (e.g. rows of
``heatmap_top_ann.txt`` after ``merge``) depends on this, so every ordering of
character keys in this package goes through :func:`sort_key`.

The collation is a process-wide setting: ``set_collation("en_US")`` (default,
matching ``LANG=en_US.UTF-8``) or ``set_collation("C")``.
"""
from __future__ import annotations

import functools
import os

_COLLATION = os.environ.get("ECOTYPER_COLLATION", "en_US")


def set_collation(locale: str) -> None:
    """Select ``"C"`` (byte order) or an ICU locale name such as ``"en_US"``."""
    global _COLLATION
    _COLLATION = locale
    _collator.cache_clear()
    sort_key.cache_clear()


def get_collation() -> str:
    return _COLLATION


@functools.lru_cache(maxsize=None)
def _collator(locale: str):
    if locale in ("C", "POSIX"):
        return None
    try:
        import icu  # PyICU
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyICU is required to reproduce R's ordering of strings in a non-C locale; "
            "install PyICU or call ecotyper.rcompat.collate.set_collation('C')"
        ) from e
    return icu.Collator.createInstance(icu.Locale(locale))


@functools.lru_cache(maxsize=1 << 20)
def sort_key(s: str):
    c = _collator(_COLLATION)
    if c is None:
        return s.encode("utf-8")
    return c.getSortKey(s)


def sorted_strings(values) -> list[str]:
    return sorted(values, key=sort_key)
