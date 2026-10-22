"""The built-in ASCII collation (used without PyICU) must order strings exactly like ICU en_US."""
import random

import pytest

from ecotyper.rcompat import collate


def _cmp(a, b):
    return (a > b) - (a < b)


def test_ascii_fallback_matches_icu():
    icu = pytest.importorskip("icu")
    c = icu.Collator.createInstance(icu.Locale("en_US"))
    rng = random.Random(1)
    alpha = [chr(i) for i in range(32, 127)]
    biased = "aAbB01._- XxZz"
    for trial in range(100_000):
        pool = biased if trial % 2 else alpha
        a = "".join(rng.choice(pool) for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice(pool) for _ in range(rng.randint(0, 8)))
        assert _cmp(collate._ascii_key(a), collate._ascii_key(b)) == c.compare(a, b), (a, b)


def test_ascii_table_matches_installed_icu():
    icu = pytest.importorskip("icu")
    c = icu.Collator.createInstance(icu.Locale("en_US"))
    for i, (p, s, t) in collate._ASCII_CE.items():
        e = c.createCollationElementIterator(chr(i)).next()
        assert (e >> 16 & 0xFFFF, e >> 8 & 0xFF, e & 0xFF) == (p, s, t), chr(i)


def test_without_icu_ascii_sorts_and_non_ascii_raises(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "icu", None)  # make `import icu` fail
    collate.set_collation("en_US")
    try:
        names = ["B.cells", "b.cells", "CD8.T.cells", "CD4.T.cells", "S01", "S1", "S_1", "IS10", "IS2"]
        out = collate.sorted_strings(names)
        assert out == sorted(names, key=collate._ascii_key)
        with pytest.raises(ImportError, match="non-ASCII"):
            collate.sort_key("Zellen_ä")
    finally:
        collate.set_collation(collate.get_collation())
