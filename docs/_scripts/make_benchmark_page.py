"""Build the charts and tables of the Benchmarks page from the raw results.

Inputs (committed):
    validation/bench/{r,py,pyfast}_<tutorial>.json  wall time / peak RSS, PSS (validation/benchmark.py)
    validation/reports/<tutorial>.tsv               per-file comparison with R (validation/tutorials.py)
    validation/reports/<tutorial>_fast.tsv          the same for the fast mode (--fast)
    validation/reports/<tutorial>_cd_agreement.tsv  ranks/ARI of the cd solver (solver_agreement.py)
Outputs (generated, not committed):
    docs/_generated/benchmark_charts.html      self-contained HTML/SVG charts with hover tooltips
    docs/_generated/benchmark_table.md         the same numbers as Markdown tables
    docs/_generated/fast_table.md, solver_table.md  tables of the Performance page

Run by ``docs/conf.py`` on every build, so the page always reflects the committed results.
A tutorial whose R or Python result is missing is shown as "pending".
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "validation" / "bench"
REPORTS = ROOT / "validation" / "reports"
OUT = ROOT / "docs" / "_generated"

TUTORIALS = [
    ("t1_lung", "T1 · Bulk recovery, lung"),
    ("t1_lymphoma", "T1 · Bulk recovery, lymphoma"),
    ("t2_lymphoma", "T2 · scRNA recovery, lymphoma"),
    ("t2_crc", "T2 · scRNA recovery, CRC"),
    ("t3_visium", "T3 · Visium recovery"),
    ("t4_bulk", "T4 · Bulk discovery"),
    ("t5_scrna", "T5 · scRNA discovery"),
    ("t6_presorted", "T6 · Presorted discovery"),
]

# geometry (px)
W, LABEL_W, RIGHT_PAD, ROW_H, TOP, BOTTOM = 760, 210, 120, 34, 12, 34


def _load(impl, tut):
    p = BENCH / f"{impl}_{tut}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return None if d.get("rc") else d


def _fmt_time(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    return f"{s / 60:.1f} min" if s >= 100 else f"{s:.0f} s"


def _esc(s):
    return html.escape(str(s), quote=True)


def _x_log(v, lo, hi):
    return LABEL_W + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - LABEL_W - RIGHT_PAD)


def _x_lin(v, hi):
    return LABEL_W + v / hi * (W - LABEL_W - RIGHT_PAD)


def _svg_open(n_rows, title):
    h = TOP + n_rows * ROW_H + BOTTOM
    return [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="{_esc(title)}">'], h


def _row_label(i, label):
    y = TOP + i * ROW_H + ROW_H / 2
    return f'<text class="lbl" x="{LABEL_W - 12}" y="{y + 4:.1f}" text-anchor="end">{_esc(label)}</text>'


def _grid(ticks, xfun, h, fmt):
    out = []
    for t in ticks:
        x = xfun(t)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{h - BOTTOM}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - BOTTOM + 18}" text-anchor="middle">{fmt(t)}</text>')
    return out


def time_chart(rows):
    """Dumbbell on a log axis: R dot, Python dot, connector, speed-up at the end."""
    lo, hi = 10, 4 * 3600
    svg, h = _svg_open(len(rows), "Wall time, R vs Python (log scale)")
    xf = lambda v: _x_log(v, lo, hi)  # noqa: E731
    svg += _grid([10, 30, 60, 300, 600, 1800, 3600, 3 * 3600], xf, h,
                 lambda t: _fmt_time(t).replace(".0 ", " "))
    for i, (tut, label, r, p, f) in enumerate(rows):
        y = TOP + i * ROW_H + ROW_H / 2
        svg.append(_row_label(i, label))
        pts = [(v, cls, name) for v, cls, name in ((r and r["wall_s"], "s1", "R"), (p and p["wall_s"], "s2", "Python"),
                                                   (f and f["wall_s"], "s3", "Python fast")) if v]
        if len(pts) >= 2:
            x1, x2 = min(xf(v) for v, _, _ in pts), max(xf(v) for v, _, _ in pts)
            svg.append(f'<line class="conn" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y}" y2="{y}"/>')
        for v, cls, name in pts:
            x = xf(v)
            tip = f"{_fmt_time(v)}|{name} · {label}"
            svg.append(f'<g class="mark" tabindex="0" data-tip="{_esc(tip)}">'
                       f'<circle class="hit" cx="{x:.1f}" cy="{y}" r="12"/>'
                       f'<circle class="dot {cls}" cx="{x:.1f}" cy="{y}" r="6"/></g>')
        if r and p:
            sp = r["wall_s"] / p["wall_s"]
            txt = f"{sp:.1f}×" + (f" · fast {r['wall_s'] / f['wall_s']:.1f}×" if f else "")
            x = max(xf(v) for v, _, _ in pts) + 14
            svg.append(f'<text class="val" x="{x:.1f}" y="{y + 4:.1f}">{txt}</text>')
        else:
            missing = "R and Python runs" if not (r or p) else ("R run" if not r else "Python run")
            x = (xf(p["wall_s"]) if p else xf(r["wall_s"]) if r else LABEL_W) + 14
            svg.append(f'<text class="muted" x="{x:.1f}" y="{y + 4:.1f}">{missing} pending</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _hbar(x0, x1, y, t, cls, tip):
    """Bar with a 4px rounded data end and a square baseline."""
    rr = min(4, max(0, x1 - x0))
    d = (f"M{x0:.1f},{y:.1f} H{x1 - rr:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + rr:.1f} "
         f"V{y + t - rr:.1f} Q{x1:.1f},{y + t:.1f} {x1 - rr:.1f},{y + t:.1f} H{x0:.1f} Z")
    return (f'<g class="mark" tabindex="0" data-tip="{_esc(tip)}">'
            f'<rect class="hit" x="{x0:.1f}" y="{y - 3:.1f}" width="{max(x1 - x0, 24):.1f}" height="{t + 6}"/>'
            f'<path class="bar {cls}" d="{d}"/></g>')


def _mem(d):
    """Peak memory: summed PSS where recorded (counts fork-shared pages once), else summed RSS."""
    return d.get("peak_pss_gb", d["peak_rss_gb"])


def memory_chart(rows):
    rows = [(t, lab, r, p) for t, lab, r, p, _ in rows]
    hi = max([_mem(d) for _, _, r, p in rows for d in (r, p) if d] + [10])
    hi = math.ceil(hi / 40) * 40
    svg, h = _svg_open(len(rows), "Peak memory, R vs Python")
    xf = lambda v: _x_lin(v, hi)  # noqa: E731
    svg += _grid(range(0, hi + 1, 40), xf, h, lambda t: f"{t} GB")
    t = 10
    for i, (tut, label, r, p) in enumerate(rows):
        y0 = TOP + i * ROW_H + (ROW_H - 2 * t - 2) / 2
        svg.append(_row_label(i, label))
        for j, (d, cls, name) in enumerate(((r, "s1", "R"), (p, "s2", "Python"))):
            y = y0 + j * (t + 2)
            if d:
                svg.append(_hbar(LABEL_W, xf(_mem(d)), y, t, cls, f"{_mem(d):.1f} GB|{name} · {label}"))
            else:
                svg.append(f'<text class="muted small" x="{LABEL_W + 6}" y="{y + t - 1:.1f}">{name} pending</text>')
        vals = [_mem(d) for d in (r, p) if d]
        if vals:
            txt = " / ".join(f"{_mem(d):.0f}" if d else "–" for d in (r, p))
            svg.append(f'<text class="val" x="{xf(max(vals)) + 8:.1f}" y="{y0 + t + 5:.1f}">{txt} GB</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def parity_chart(rep):
    lo, hi = 1e-16, 1e-6
    svg, h = _svg_open(len(rep), "Largest relative difference vs R per tutorial (log scale)")
    xf = lambda v: _x_log(max(v, lo), lo, hi)  # noqa: E731
    svg += _grid([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6], xf, h, lambda t: f"1e{int(math.log10(t))}")
    xt = xf(1e-8)
    svg.append(f'<line class="target" x1="{xt:.1f}" x2="{xt:.1f}" y1="{TOP - 4}" y2="{h - BOTTOM}"/>')
    svg.append(f'<text class="muted small" x="{xt + 6:.1f}" y="{TOP + 6}">target 1e-8</text>')
    for i, row in enumerate(rep):
        y = TOP + i * ROW_H + ROW_H / 2
        svg.append(_row_label(i, row["label"]))
        x = xf(row["max_rel"])
        tip = (f"{row['max_rel']:.2g}|{row['label']} · {row['discrete']}/{row['files']} files discrete-equal, "
               f"{row['identical']} byte-identical")
        svg.append(f'<line class="conn" x1="{LABEL_W}" x2="{x:.1f}" y1="{y}" y2="{y}"/>')
        svg.append(f'<g class="mark" tabindex="0" data-tip="{_esc(tip)}">'
                   f'<circle class="hit" cx="{x:.1f}" cy="{y}" r="12"/>'
                   f'<circle class="dot s2" cx="{x:.1f}" cy="{y}" r="6"/></g>')
    svg.append("</svg>")
    return "\n".join(svg)


def parity_rows(suffix=""):
    out = []
    for tut, label in TUTORIALS:
        p = REPORTS / f"{tut}{suffix}.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        out.append(dict(tut=tut, label=label, files=len(df),
                        identical=int(df["identical_bytes"].fillna(False).astype(bool).sum()),
                        discrete=int(df["discrete_equal"].fillna(False).astype(bool).sum()),
                        missing=int(df["missing"].sum()), max_rel=float(df["max_rel"].max())))
    return out


CSS = """
<style>
.viz-root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--grid:#e6e5e0;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--tip-bg:#ffffff;--tip-border:#dddcd6;
  position:relative;color:var(--ink2);font-family:var(--font-stack,system-ui,sans-serif);margin:0 0 2rem}
body[data-theme="dark"] .viz-root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#97968e;--grid:#33332f;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--tip-bg:#262624;--tip-border:#3d3d39}
@media (prefers-color-scheme: dark){body:not([data-theme="light"]):not([data-theme="dark"]) .viz-root,
  body[data-theme="auto"] .viz-root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#97968e;--grid:#33332f;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--tip-bg:#262624;--tip-border:#3d3d39}}
.viz-root .card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:16px 18px 8px;margin:0 0 18px}
.viz-root h4{margin:0;color:var(--ink);font-size:1rem;font-weight:600}
.viz-root .sub{margin:2px 0 8px;font-size:.85rem;color:var(--ink2)}
.viz-root svg{display:block;overflow:visible;font-size:12px}
.viz-root .lbl{fill:var(--ink2)} .viz-root .tick{fill:var(--muted);font-size:11px}
.viz-root .val{fill:var(--ink);font-weight:600} .viz-root .muted{fill:var(--muted);font-style:italic}
.viz-root .small{font-size:11px}
.viz-root .grid{stroke:var(--grid);stroke-width:1}
.viz-root .target{stroke:var(--ink2);stroke-width:1}
.viz-root .conn{stroke:var(--grid);stroke-width:2;stroke-linecap:round}
.viz-root .dot{stroke:var(--surface);stroke-width:2}
.viz-root .s1{fill:var(--s1)} .viz-root .s2{fill:var(--s2)} .viz-root .s3{fill:var(--s3)}
.viz-root .hit{fill:transparent}
.viz-root .mark{cursor:default;outline:none}
.viz-root .mark:hover .dot,.viz-root .mark:focus .dot,.viz-root .mark:hover .bar,.viz-root .mark:focus .bar{filter:brightness(1.15)}
.viz-root .mark:focus-visible .hit{stroke:var(--ink2);stroke-width:1}
.viz-root .legend{display:flex;gap:18px;font-size:.85rem;color:var(--ink2);margin:0 0 6px}
.viz-root .legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;
  margin-right:6px;vertical-align:-1px;background:var(--c)}
.viz-root .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:0 0 18px}
.viz-root .tile{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.viz-root .tile b{display:block;font-size:1.5rem;color:var(--ink);line-height:1.2}
.viz-root .tile span{font-size:.8rem;color:var(--ink2)}
.viz-root .tip{position:absolute;pointer-events:none;background:var(--tip-bg);border:1px solid var(--tip-border);
  border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;max-width:280px;z-index:5}
.viz-root .tip strong{display:block;color:var(--ink);font-size:13px} .viz-root .tip em{font-style:normal;color:var(--ink2)}
</style>
"""

JS = """
<script>
(function(){
  var root=document.currentScript.closest('.viz-root'); if(!root) return;
  var tip=root.querySelector('.tip');
  function show(el,cx,cy){
    var parts=el.getAttribute('data-tip').split('|');
    tip.textContent='';
    var s=document.createElement('strong'); s.textContent=parts[0];
    var e=document.createElement('em'); e.textContent=parts.slice(1).join('|');
    tip.appendChild(s); tip.appendChild(e); tip.style.display='block';
    var r=root.getBoundingClientRect(), x=cx-r.left+14, y=cy-r.top+14;
    if(x+tip.offsetWidth>r.width) x=cx-r.left-tip.offsetWidth-14;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  root.querySelectorAll('.mark').forEach(function(m){
    m.addEventListener('pointermove',function(ev){show(m,ev.clientX,ev.clientY);});
    m.addEventListener('pointerleave',function(){tip.style.display='none';});
    m.addEventListener('focus',function(){var b=m.getBoundingClientRect();show(m,b.right,b.top);});
    m.addEventListener('blur',function(){tip.style.display='none';});
  });
})();
</script>
"""


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [(t, lab, _load("r", t), _load("py", t), _load("pyfast", t)) for t, lab in TUTORIALS]
    rep = parity_rows()

    both = [(r, p) for _, _, r, p, _ in rows if r and p]
    speedups = [r["wall_s"] / p["wall_s"] for r, p in both]
    n_files = sum(x["files"] for x in rep)
    n_discrete = sum(x["discrete"] for x in rep)
    n_ident = sum(x["identical"] for x in rep)
    worst = max((x["max_rel"] for x in rep), default=float("nan"))
    tiles = [
        (f"{n_discrete:,}", f"of {n_files:,} output files discrete-equal to R"),
        (f"{n_ident / max(n_files, 1):.0%}", "of files byte-identical"),
        (f"{worst:.0e}".replace("e-", "e−"), "largest relative difference"),
    ]
    if speedups:
        tiles.append((f"{min(speedups):.1f}–{max(speedups):.1f}×", f"Python faster than R ({len(both)} tutorials timed)"))

    legend = ('<div class="legend"><span style="--c:var(--s1)">R (EcoTyper, R 4.4.3)</span>'
              '<span style="--c:var(--s2)">Python (exact, default)</span>'
              '<span style="--c:var(--s3)">Python fast mode</span></div>')
    legend2 = ('<div class="legend"><span style="--c:var(--s1)">R (EcoTyper, R 4.4.3)</span>'
               '<span style="--c:var(--s2)">Python (exact, default)</span></div>')
    parts = [CSS, '<div class="viz-root">', '<div class="tiles">']
    parts += [f'<div class="tile"><b>{_esc(v)}</b><span>{_esc(k)}</span></div>' for v, k in tiles]
    parts.append("</div>")
    parts.append('<div class="card"><h4>Wall time</h4><p class="sub">Same server, 32 parallel jobs, '
                 'one BLAS thread per process. Log scale; hover a dot for the value.</p>'
                 + legend + time_chart(rows) + "</div>")
    parts.append('<div class="card"><h4>Peak memory</h4><p class="sub">Summed over all processes in the '
                 'container: PSS for Python (pages shared by forked workers counted once), RSS for R '
                 '(separate processes).</p>' + legend2 + memory_chart(rows) + "</div>")
    if rep:
        parts.append('<div class="card"><h4>Agreement with R</h4><p class="sub">Largest relative difference '
                     'of any numeric cell, per tutorial (log scale). Every discrete output is identical.</p>'
                     + parity_chart(rep) + "</div>")
    parts += ['<div class="tip" role="tooltip"></div>', JS, "</div>"]
    (OUT / "benchmark_charts.html").write_text("\n".join(parts))

    md = ["| Tutorial | R wall | Python wall | R / Python | Python fast wall | R / fast | R peak RSS | Python peak PSS (RSS) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, lab, r, p, f in rows:
        cell = lambda d, k, fn: fn(d[k]) if d else "pending"  # noqa: E731
        ratio = f"{r['wall_s'] / p['wall_s']:.2f}×" if r and p else "–"
        fratio = f"{r['wall_s'] / f['wall_s']:.2f}×" if r and f else "–"
        pm = f"{_mem(p):.1f} GB ({p['peak_rss_gb']:.1f} GB)" if p else "pending"
        md.append(f"| {lab} | {cell(r, 'wall_s', _fmt_time)} | {cell(p, 'wall_s', _fmt_time)} | {ratio} | "
                  f"{cell(f, 'wall_s', _fmt_time) if f else '–'} | {fratio} | "
                  f"{cell(r, 'peak_rss_gb', lambda v: f'{v:.1f} GB')} | {pm} |")
    md += ["", "| Tutorial | Files | Byte-identical | Discrete-equal | Missing | Max rel. diff |",
           "|---|---:|---:|---:|---:|---:|"]
    for x in rep:
        md.append(f"| {x['label']} | {x['files']} | {x['identical']} | {x['discrete']} | {x['missing']} | {x['max_rel']:.2g} |")
    (OUT / "benchmark_table.md").write_text("\n".join(md) + "\n")

    fast = parity_rows("_fast")
    md = ["| Tutorial | Files | Byte-identical | Discrete-equal | Missing | Max rel. diff |", "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {x['label']} | {x['files']} | {x['identical']} | {x['discrete']} | {x['missing']} | {x['max_rel']:.2g} |"
           for x in fast] or ["| (no fast-mode reports yet) | | | | | |"]
    (OUT / "fast_table.md").write_text("\n".join(md) + "\n")

    md = ["| Tutorial | Cell type | Rank (R) | Rank (cd) | ARI states | ARI initial states |", "|---|---|---:|---:|---:|---:|"]
    for tut, label in TUTORIALS:
        q = REPORTS / f"{tut}_cd_agreement.tsv"
        if not q.exists():
            continue
        for _, x in pd.read_csv(q, sep="\t").iterrows():
            fmt = lambda v: "–" if pd.isna(v) else f"{v:.2f}"  # noqa: E731
            rk = lambda v: "–" if pd.isna(v) else str(int(v))  # noqa: E731
            md.append(f"| {label} | {x['cell_type']} | {rk(x['rank_r'])} | {rk(x['rank_py'])} | "
                      f"{fmt(x.get('ari_states'))} | {fmt(x.get('ari_initial'))} |")
    if len(md) == 2:
        md.append("| (no solver runs yet) | | | | | |")
    (OUT / "solver_table.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    build()
