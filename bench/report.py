"""Summarizes a bench/run.sh CSV: median over rounds, guest vs native side by side."""
import csv, statistics, sys, platform, subprocess, datetime
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
vals = defaultdict(list)
units = {}
for r in rows:
    key = (r["bench"], r["size"])
    vals[(r["env"],) + key].append((float(r["gpu_ms"]), float(r["wall_ms"]), float(r["value"])))
    units[key] = r["unit"]
rounds = len({r["round"] for r in rows})

def med(env, key, i):
    v = vals.get((env,) + key)
    return statistics.median(x[i] for x in v) if v else None

def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception: return "?"

higher_better = {"GB/s", "TFLOP/s", "dispatch/s"}
print("# MetalPassthrough benchmark results\n")
print(f"{datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC · {sh('sysctl -n machdep.cpu.brand_string')} · macOS {sh('sw_vers -productVersion')} "
      f"· median of {rounds} alternating rounds (native, guest, native, …) × the per-run repetitions\n")
print("GPU = command-buffer GPU timestamps. Wall = from the API call until the result is visible to the caller "
      "(native: CPU in the same process; guest: CPU inside the Linux VM).\n")
print("| Benchmark | Size | Native GPU ms | Guest GPU ms | Native wall ms | Guest wall ms | Native | Guest | Guest / native |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
import re
order = list(dict.fromkeys(r["bench"] for r in rows))
def size_key(sz):
    m = re.match(r"(\d+)(KiB|MiB)?", sz)
    return int(m.group(1)) * {"KiB": 1, "MiB": 1024}.get(m.group(2), 1) if m else 0
keys = sorted({k[1:] for k in vals if k[0] == "native"}, key=lambda k: (order.index(k[0]), size_key(k[1])))
for key in keys:
    u = units[key]
    n_g, g_g = med("native", key, 0), med("guest", key, 0)
    n_w, g_w = med("native", key, 1), med("guest", key, 1)
    n_v, g_v = med("native", key, 2), med("guest", key, 2)
    if g_v is None: continue
    ratio = (g_v / n_v) if u in higher_better else (n_v / g_v if g_v else 0)
    fmt = lambda x: "—" if not x else f"{x:.3f}"
    print(f"| {key[0]} | {key[1]} | {fmt(n_g)} | {fmt(g_g)} | {fmt(n_w)} | {fmt(g_w)} | {n_v:.2f} {u} | {g_v:.2f} {u} | {ratio * 100:.0f}% |")
print("\nGuest-only measurements:\n")
print("| Measurement | Size | Value |")
print("|---|---|---:|")
for k in sorted({k[1:] for k in vals if k[0] == "guest"} - set(keys)):
    print(f"| {k[0]} | {k[1]} | {med('guest', k, 2):.2f} {units[k]} |")
print("\n*Guest / native* is relative performance (100% = native speed; for latencies it is native time ÷ guest time).")
