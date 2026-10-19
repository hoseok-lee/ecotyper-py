"""Benchmark R EcoTyper against the Python port, on the same machine with the same job count.

Run on the compute server (needs work/dock.sh, work/dock_r.sh and the two docker images):

    python validation/benchmark.py --jobs 32 t1_lung t1_lymphoma t2_lymphoma t2_crc t3_visium t6_presorted t4_bulk

Each run starts in a fresh container, one BLAS/OpenMP thread per process, parallelism from the job
queue (R: `Number of threads`, Python: `n_jobs`). Python timings include numba JIT warm-up (cold
cache). Results go to validation/bench/<impl>_<tutorial>.json and are summarised with `--table`.
Reference outputs are not touched: R writes into work/ecotyper-r-src, Python into work/bench_py/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "validation" / "bench"
ORDER = ["t1_lung", "t1_lymphoma", "t2_lymphoma", "t2_crc", "t3_visium", "t6_presorted", "t4_bulk", "t5_scrna"]


def run_impl(impl, tut, jobs):
    BENCH.mkdir(parents=True, exist_ok=True)
    out = f"validation/bench/{impl}_{tut}.json"
    if impl == "r":
        env = dict(os.environ, ECOTYPER_R_THREADS=str(jobs))
        cmd = ["work/dock_r.sh", "python3", "validation/bench_wrap.py", f"r_{tut}", out, "--",
               "bash", "validation/r_reference/run_r_reference.sh", tut]
    else:
        env = dict(os.environ, ECOTYPER_NO_FMA="1", ECOTYPER_R_REF="/work/work/r_ref_x86")
        shutil.rmtree(ROOT / "work" / "bench_py" / tut, ignore_errors=True)
        cmd = ["work/dock.sh", "python", "validation/bench_wrap.py", f"py_{tut}", out, "--",
               "python", "validation/tutorials.py", tut, "--out", f"work/bench_py/{tut}",
               "--jobs", str(jobs), "--no-compare"]
    log = ROOT / "work" / "logs" / f"bench_{impl}_{tut}.log"
    with open(log, "w") as f:
        subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def fmt(s):
    return f"{s / 60:.1f} min" if s >= 100 else f"{s:.0f} s"


def table():
    rows = ["| Tutorial | R wall | Python wall | Speed-up (R/Py) | R peak RSS | Python peak RSS |",
            "|---|---|---|---|---|---|"]
    for t in ORDER:
        try:
            r = json.load(open(BENCH / f"r_{t}.json"))
            p = json.load(open(BENCH / f"py_{t}.json"))
        except FileNotFoundError:
            continue
        if r["rc"] or p["rc"]:
            continue
        rows.append(f"| {t} | {fmt(r['wall_s'])} | {fmt(p['wall_s'])} | {r['wall_s'] / p['wall_s']:.2f}x | "
                    f"{r['peak_rss_gb']} GB | {p['peak_rss_gb']} GB |")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tutorials", nargs="*")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--impl", choices=["r", "py", "both"], default="both")
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
