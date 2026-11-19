#!/bin/sh
# Reproducible benchmark: identical kernels/sizes natively on the host and from a Linux guest through
# MetalPassthrough, alternating for ROUNDS rounds (so system noise hits both sides equally).
# Writes bench/results/<timestamp>.csv and bench/results/RESULTS.md.
set -eu
cd "$(dirname "$0")/.."
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-7}
RUNTIME=${RUNTIME:-metalpassthrough-runtime:latest}
./scripts/build-host.sh release >/dev/null
BIN=$(swift build -c release --show-bin-path)
container run --rm -v "$PWD:/src" -w /src/bench metalpassthrough-builder:latest \
    gcc -O2 -o bench_guest bench_guest.c -I/src/guest/dist/include -L/src/guest/dist/lib -lmetalpassthrough -lm \
    -Wl,-rpath,/usr/local/metalpassthrough/lib >/dev/null 2>&1
STAMP=$(date -u +%Y%m%d-%H%M%S)
CSV=bench/results/$STAMP.csv
echo "round,env,bench,size,gpu_ms,wall_ms,value,unit" > "$CSV"
for r in $(seq 1 "$ROUNDS"); do
    echo "round $r/$ROUNDS: native" >&2
    "$BIN/mpt-bench" bench/kernels.metal "$REPS" | sed "s/^/$r,/" >> "$CSV"
    echo "round $r/$ROUNDS: guest" >&2
    TMP=$(mktemp)
    "$BIN/mpt-run" --cpus 4 --memory 4096 --hugepages 1536 --runtime "$PWD/guest/dist" -v "$PWD/bench:/bench" \
        "$RUNTIME" /bench/bench_guest /bench/kernels.metal "$REPS" > "$TMP" 2>/dev/null &
    PID=$!
    PEAK=0
    while kill -0 $PID 2>/dev/null; do  # host-side resident memory of the VM's GPU service process
        RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ' || echo 0)
        [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
        sleep 0.5
    done
    wait $PID || true
    grep '^guest,' "$TMP" | sed "s/^/$r,/" >> "$CSV"
    echo "$r,guest,host_rss_peak,mpt-run,0,0,$((PEAK / 1024)),MiB" >> "$CSV"
    rm -f "$TMP"
done
python3 bench/report.py "$CSV" > bench/results/RESULTS.md
echo "wrote $CSV and bench/results/RESULTS.md" >&2
