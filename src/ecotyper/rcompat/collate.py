"""String collation as used by R's ``sort``/``order``/``factor``/``merge``.

In a non-C locale R (built with ICU, as the conda-forge and CRAN macOS builds
are) compares strings with an ICU collator for the current locale; in the C
locale it compares bytes. EcoTyper's output ordering (e.g. rows of
``heatmap_top_ann.txt`` after ``merge``) depends on this, so every ordering of
character keys in this package goes through :func:`sort_key`.

The collation is a process-wide setting: ``set_collation("en_US")`` (default,
matching ``LANG=en_US.UTF-8``) or ``set_collation("C")``.

PyICU is optional. Without it, strings made of printable ASCII (all gene, cell
type, state and sample names in practice) are ordered with a built-in copy of
ICU's ``en_US`` collation weights for those 95 characters: every ASCII character
maps to one collation element (no contractions), so comparing primary, then
secondary, then tertiary weights reproduces ICU exactly (checked against
PyICU/ICU 78 on 200,000 random string pairs, see ``tests/test_collate.py``).
Other strings need PyICU (``pip install "ecotyper-py[icu]"``).
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


# ICU en_US collation element of each printable ASCII character: (primary, secondary, tertiary)
_ASCII_CE = {
    32: (0x0400, 0x05, 0x05),  # ' '
    33: (0x075a, 0x05, 0x05),  # '!'
    34: (0x0976, 0x05, 0x05),  # '"'
    35: (0x0a94, 0x05, 0x05),  # '#'
    36: (0x0da9, 0x05, 0x05),  # '$'
    37: (0x0a96, 0x05, 0x05),  # '%'
    38: (0x0a8e, 0x05, 0x05),  # '&'
    39: (0x0970, 0x05, 0x05),  # "'"
    40: (0x097c, 0x05, 0x05),  # '('
    41: (0x097e, 0x05, 0x05),  # ')'
    42: (0x0a7e, 0x05, 0x05),  # '*'
    43: (0x0c7b, 0x05, 0x05),  # '+'
    44: (0x0600, 0x05, 0x05),  # ','
    45: (0x050e, 0x05, 0x05),  # '-'
    46: (0x0800, 0x05, 0x05),  # '.'
    47: (0x0a88, 0x05, 0x05),  # '/'
    48: (0x1200, 0x05, 0x05),  # '0'
    49: (0x1400, 0x05, 0x05),  # '1'
    50: (0x1600, 0x05, 0x05),  # '2'
    51: (0x1800, 0x05, 0x05),  # '3'
    52: (0x1a00, 0x05, 0x05),  # '4'
    53: (0x1c00, 0x05, 0x05),  # '5'
    54: (0x1e00, 0x05, 0x05),  # '6'
    55: (0x2000, 0x05, 0x05),  # '7'
    56: (0x2200, 0x05, 0x05),  # '8'
    57: (0x2400, 0x05, 0x05),  # '9'
    58: (0x0736, 0x05, 0x05),  # ':'
    59: (0x072a, 0x05, 0x05),  # ';'
    60: (0x0c85, 0x05, 0x05),  # '<'
    61: (0x0c87, 0x05, 0x05),  # '='
    62: (0x0c89, 0x05, 0x05),  # '>'
    63: (0x0764, 0x05, 0x05),  # '?'
    64: (0x0a7c, 0x05, 0x05),  # '@'
    65: (0x2b00, 0x05, 0x9c),  # 'A'
    66: (0x2d00, 0x05, 0x9c),  # 'B'
    67: (0x2f00, 0x05, 0x9c),  # 'C'
    68: (0x3100, 0x05, 0x9c),  # 'D'
    69: (0x3300, 0x05, 0x9c),  # 'E'
    70: (0x3500, 0x05, 0x9c),  # 'F'
    71: (0x3700, 0x05, 0x9c),  # 'G'
    72: (0x3900, 0x05, 0x9c),  # 'H'
    73: (0x3b00, 0x05, 0x9c),  # 'I'
    74: (0x3d00, 0x05, 0x9c),  # 'J'
    75: (0x3f00, 0x05, 0x9c),  # 'K'
    76: (0x4100, 0x05, 0x9c),  # 'L'
    77: (0x4300, 0x05, 0x9c),  # 'M'
    78: (0x4500, 0x05, 0x9c),  # 'N'
    79: (0x4700, 0x05, 0x9c),  # 'O'
    80: (0x4900, 0x05, 0x9c),  # 'P'
    81: (0x4b00, 0x05, 0x9c),  # 'Q'
    82: (0x4d00, 0x05, 0x9c),  # 'R'
    83: (0x4f00, 0x05, 0x9c),  # 'S'
    84: (0x5100, 0x05, 0x9c),  # 'T'
    85: (0x5300, 0x05, 0x9c),  # 'U'
    86: (0x5500, 0x05, 0x9c),  # 'V'
    87: (0x5700, 0x05, 0xa0),  # 'W'
    88: (0x5900, 0x05, 0x9c),  # 'X'
    89: (0x5b00, 0x05, 0x9c),  # 'Y'
    90: (0x5d00, 0x05, 0x9c),  # 'Z'
    91: (0x0980, 0x05, 0x05),  # '['
    92: (0x0a8a, 0x05, 0x05),  # '\\'
    93: (0x0982, 0x05, 0x05),  # ']'
    94: (0x0c0a, 0x05, 0x05),  # '^'
    95: (0x050a, 0x05, 0x05),  # '_'
    96: (0x0c04, 0x05, 0x05),  # '`'
    97: (0x2b00, 0x05, 0x05),  # 'a'
    98: (0x2d00, 0x05, 0x05),  # 'b'
    99: (0x2f00, 0x05, 0x05),  # 'c'
    100: (0x3100, 0x05, 0x05),  # 'd'
    101: (0x3300, 0x05, 0x05),  # 'e'
    102: (0x3500, 0x05, 0x05),  # 'f'
    103: (0x3700, 0x05, 0x05),  # 'g'
    104: (0x3900, 0x05, 0x05),  # 'h'
    105: (0x3b00, 0x05, 0x05),  # 'i'
    106: (0x3d00, 0x05, 0x05),  # 'j'
    107: (0x3f00, 0x05, 0x05),  # 'k'
    108: (0x4100, 0x05, 0x05),  # 'l'
    109: (0x4300, 0x05, 0x05),  # 'm'
    110: (0x4500, 0x05, 0x05),  # 'n'
    111: (0x4700, 0x05, 0x05),  # 'o'
    112: (0x4900, 0x05, 0x05),  # 'p'
    113: (0x4b00, 0x05, 0x05),  # 'q'
    114: (0x4d00, 0x05, 0x05),  # 'r'
    115: (0x4f00, 0x05, 0x05),  # 's'
    116: (0x5100, 0x05, 0x05),  # 't'
    117: (0x5300, 0x05, 0x05),  # 'u'
    118: (0x5500, 0x05, 0x05),  # 'v'
    119: (0x5700, 0x05, 0x05),  # 'w'
    120: (0x5900, 0x05, 0x05),  # 'x'
    121: (0x5b00, 0x05, 0x05),  # 'y'
    122: (0x5d00, 0x05, 0x05),  # 'z'
    123: (0x0984, 0x05, 0x05),  # '{'
    124: (0x0c8d, 0x05, 0x05),  # '|'
    125: (0x0986, 0x05, 0x05),  # '}'
    126: (0x0c91, 0x05, 0x05),  # '~'
}
_ASCII_P = {chr(i): w[0] for i, w in _ASCII_CE.items()}
_ASCII_S = {chr(i): w[1] for i, w in _ASCII_CE.items()}
_ASCII_T = {chr(i): w[2] for i, w in _ASCII_CE.items()}


def _ascii_key(s: str):
    return (tuple(map(_ASCII_P.__getitem__, s)), tuple(map(_ASCII_S.__getitem__, s)),
            tuple(map(_ASCII_T.__getitem__, s)))


_NO_ICU = object()


@functools.lru_cache(maxsize=None)
def _collator(locale: str):
    if locale in ("C", "POSIX"):
        return None
    try:
        import icu  # PyICU
    except ImportError:
        if locale.split(".")[0] not in ("en_US", "en", "root", ""):
            raise ImportError(
                f"PyICU is required to reproduce R's ordering of strings in locale {locale!r}; "
                'install it (pip install "ecotyper-py[icu]") or call '
                "ecotyper.rcompat.collate.set_collation('C')"
            ) from None
        return _NO_ICU
    return icu.Collator.createInstance(icu.Locale(locale))


@functools.lru_cache(maxsize=1 << 20)
def sort_key(s: str):
    c = _collator(_COLLATION)
    if c is None:
        return s.encode("utf-8")
    if c is _NO_ICU:
        try:
            return _ascii_key(s)
        except KeyError:
            raise ImportError(
                f"PyICU is required to reproduce R's ordering of the non-ASCII string {{s!r}}; "
                'install it (pip install "ecotyper-py[icu]") or call '
                "ecotyper.rcompat.collate.set_collation('C')"
            ) from None
    return c.getSortKey(s)


def sorted_strings(values) -> list[str]:
    return sorted(values, key=sort_key)
