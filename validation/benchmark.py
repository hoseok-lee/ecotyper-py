"""Benchmark R EcoTyper against the Python port, on the same machine with the same job count.

Run on the compute server (needs work/dock.sh, work/dock_r.sh and the two docker images):

    python validation/benchmark.py --jobs 32 t1_lung t1_lymphoma t2_lymphoma t2_crc t3_visium t6_presorted t4_bulk

Each run starts in a fresh container, one BLAS/OpenMP thread per process, parallelism from the job
queue (R: `Number of threads`, Python: `n_jobs`). Python timings include numba JIT warm-up (cold
cache). `--impl pyfast` runs the fast mode (`exact=False`). Results go to
validation/bench/<impl>_<tutorial>.json (wall time, peak summed RSS and PSS) and are summarised
with `--table`.
Reference outputs are not touched: R writes into work/ecotyper-r-src, Python into work/bench_py/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "validation" / "bench"
ORDER = ["t1_lung", "t1_lymphoma", "t2_lymphoma", "t2_crc", "t3_visium", "t6_presorted", "t4_bulk", "t5_scrna"]
# R discovery skips every NMF restart whose estim.RData already exists under EcoTyper/<name>, so a
# rerun would time almost nothing. Move that cache aside (kept under work/r_cache_bak) first.
R_DISCOVERY_CACHE = {"t4_bulk": "MyDiscovery", "t5_scrna": "discovery_scRNA_CRC", "t6_presorted": "PresortedDiscovery"}


def run_impl(impl, tut, jobs):
    BENCH.mkdir(parents=True, exist_ok=True)
    out = f"validation/bench/{impl}_{tut}.json"
    if impl == "r":
        cache = ROOT / "work" / "ecotyper-r-src" / "EcoTyper" / R_DISCOVERY_CACHE.get(tut, "")
        if tut in R_DISCOVERY_CACHE and cache.exists():
            bak = ROOT / "work" / "r_cache_bak"
            bak.mkdir(exist_ok=True)
            shutil.move(cache, bak / f"{cache.name}.{time.strftime('%Y%m%d-%H%M%S')}")
        env = dict(os.environ, ECOTYPER_R_THREADS=str(jobs))
        cmd = ["work/dock_r.sh", "python3", "validation/bench_wrap.py", f"r_{tut}", out, "--",
               "bash", "validation/r_reference/run_r_reference.sh", tut]
    else:  # py (exact, default) or pyfast (exact=False)
        env = dict(os.environ, ECOTYPER_NO_FMA="1", ECOTYPER_R_REF="/work/work/r_ref_x86")
        shutil.rmtree(ROOT / "work" / f"bench_{impl}" / tut, ignore_errors=True)
        cmd = ["work/dock.sh", "python", "validation/bench_wrap.py", f"{impl}_{tut}", out, "--",
               "python", "validation/tutorials.py", tut, "--out", f"work/bench_{impl}/{tut}",
               "--jobs", str(jobs), "--no-compare"] + (["--fast"] if impl == "pyfast" else [])
    log = ROOT / "work" / "logs" / f"bench_{impl}_{tut}.log"
    with open(log, "w") as f:
        subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def fmt(s):
    return f"{s / 60:.1f} min" if s >= 100 else f"{s:.0f} s"


def _load(impl, t):
    try:
        d = json.load(open(BENCH / f"{impl}_{t}.json"))
    except FileNotFoundError:
        return None
    return None if d["rc"] else d


def table():
    rows = ["| Tutorial | R wall | Python wall | R/Py | Python fast wall | R/fast | R peak RSS | Python peak PSS (RSS) |",
            "|---|---|---|---|---|---|---|---|"]
    for t in ORDER:
        r, p, f = _load("r", t), _load("py", t), _load("pyfast", t)
        if r is None or p is None:
            continue
        mem = f"{p['peak_pss_gb']} GB ({p['peak_rss_gb']} GB)" if "peak_pss_gb" in p else f"– ({p['peak_rss_gb']} GB)"
        rows.append(f"| {t} | {fmt(r['wall_s'])} | {fmt(p['wall_s'])} | {r['wall_s'] / p['wall_s']:.2f}x | "
                    + (f"{fmt(f['wall_s'])} | {r['wall_s'] / f['wall_s']:.2f}x | " if f else "– | – | ")
                    + f"{r['peak_rss_gb']} GB | {mem} |")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tutorials", nargs="*")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--impl", choices=["r", "py", "pyfast", "both"], default="both")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table()
    for t in a.tutorials:
        for impl in (["r", "py"] if a.impl == "both" else [a.impl]):
            run_impl(impl, t, a.jobs)
            print(f"done {impl} {t}", flush=True)


if __name__ == "__main__":
    main()
