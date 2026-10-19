"""Run a command, sample total RSS of all other processes in this container, print one JSON line.

Usage (inside a container): python3 bench_wrap.py LABEL OUTJSON -- cmd args...
Peak RSS is the maximum over 0.5 s samples of the summed VmRSS of every process except this one
(the container only holds the benchmarked command and its workers; shared pages are counted per
process, so the total is an upper bound for fork-heavy workloads).
"""
import json
import os
import subprocess
import sys
import threading
import time


def total_rss_kb(me):
    tot = 0
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
                        break
        except OSError:
            pass
    return tot


def main():
    label, outjson = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    me = os.getpid()
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], total_rss_kb(me))
            stop.wait(0.5)

    th = threading.Thread(target=sample, daemon=True)
    t0 = time.time()
    th.start()
    rc = subprocess.call(cmd)
    wall = time.time() - t0
    stop.set()
    th.join()
    rec = {"label": label, "rc": rc, "wall_s": round(wall, 1), "peak_rss_gb": round(peak[0] / 1048576, 2)}
    with open(outjson, "w") as f:
        json.dump(rec, f)
    print(json.dumps(rec))
    sys.exit(rc)


if __name__ == "__main__":
    main()
