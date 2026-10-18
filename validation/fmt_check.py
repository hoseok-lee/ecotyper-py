"""Compare rcompat.io.format_real with R's write.table on ~460k doubles.
usage: fmt_check.py <dir containing fmt_x.bin / fmt_r.txt (from fmt_check.R)>"""
import sys
from pathlib import Path

import numpy as np

from ecotyper.rcompat.io import format_real

d = Path(sys.argv[1])
x = np.fromfile(d / "fmt_x.bin", dtype="<f8")
r = (d / "fmt_r.txt").read_text().split("\n")[:-1]
assert len(r) == len(x)
bad = [(v, a, format_real(float(v))) for v, a in zip(x, r) if format_real(float(v)) != a]
print(f"fmt_check: {len(x)} values, {len(bad)} differ")
for v, a, b in bad[:10]:
    print(f"  {v!r}: R={a} py={b}")
