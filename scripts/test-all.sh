#!/bin/sh
# Runs the MetalPassthrough test suite: host unit tests + the four end-to-end tests, each inside a fresh
# Linux container (apple/containerization VM) on this Mac. Logs land in build/results/.
# Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=build/results
mkdir -p "$OUT"
RUNTIME=${RUNTIME:-metalpassthrough-runtime:latest}
pass=0; fail=0
results=""

step() {  # name, logfile, command...
    name=$1; log=$2; shift 2
    printf '%-58s' "$name"
    if "$@" > "$OUT/$log" 2>&1; then
        echo "PASS"; pass=$((pass + 1)); results="$results\n| $name | PASS | \`$log\` |"
    else
        echo "FAIL (see $OUT/$log)"; fail=$((fail + 1)); results="$results\n| $name | **FAIL** | \`$log\` |"
    fi
}

echo "== build"
step "host: swift build -c release + codesign" build-host.log ./scripts/build-host.sh release
MPT="$(swift build -c release --show-bin-path)/mpt-run"
step "guest: libmetalpassthrough, libcuda, libcudart, apple-smi" build-guest.log make guest
step "e2e programs (gcc + nvcc, Linux arm64)" build-e2e.log make e2e

run() {  # extra mpt-run args ... -- command
    "$MPT" --cpus 8 --memory 8192 --hugepages 2048 --runtime "$ROOT/guest/dist" -v "$ROOT/e2e:/e2e" "$@"
}

echo "== unit"
step "PTX->MSL translator + Metal execution (swift test)" unit.log swift test

echo "== end-to-end (inside Linux containers)"
step "selftest: ring, allocator, zero-copy SAXPY" selftest.log run "$RUNTIME" mpt-selftest
step "Test 1: apple-smi hardware detection" test1-apple-smi.log run "$RUNTIME" sh -c '
    apple-smi && apple-smi --json > /tmp/smi.json && python3 -c "
import json; d = json.load(open(\"/tmp/smi.json\")); print(d)
assert d[\"name\"].startswith(\"Apple\") and d[\"gpu_cores\"] > 0 and d[\"unified_memory_bytes\"] > 0
assert d[\"ring_latency_ns\"][\"median\"] < 100000
print(\"apple-smi: PASS\")"'
step "Test 2: zero-copy native Metal (256 MiB, bit-exact)" test2-native.log run -e METALPASSTHROUGH_ARENA_MB=1100 "$RUNTIME" /e2e/native/native_test
step "Test 3a: CUDA Driver API + nvcc PTX" test3-driver.log run "$RUNTIME" /e2e/cuda/driver_test /e2e/cuda/kernels.ptx 1024
step "Test 3b: nvcc-built CUDA Runtime executable" test3-runtime.log run "$RUNTIME" /e2e/cuda/runtime_test
step "Test 3c: CUDA streams, events, async copies, bounds" test3-semantics.log run "$RUNTIME" /e2e/cuda/semantics_test
step "Test 4: Python NumPy/PyTorch zero-copy" test4-python.log run -e METALPASSTHROUGH_ARENA_MB=1024 "$RUNTIME" python3 /e2e/python/test4_numpy_torch.py

echo "== translated addressing (UVA negotiated off by the host)"
step "no-UVA: selftest" nouva-selftest.log env METALPASSTHROUGH_NO_UVA=1 "$MPT" --hugepages 1024 --runtime "$ROOT/guest/dist" "$RUNTIME" mpt-selftest
step "no-UVA: CUDA Driver API" nouva-driver.log env METALPASSTHROUGH_NO_UVA=1 "$MPT" --hugepages 2048 --memory 8192 --runtime "$ROOT/guest/dist" -v "$ROOT/e2e:/e2e" "$RUNTIME" /e2e/cuda/driver_test /e2e/cuda/kernels.ptx 512
step "no-UVA: CUDA semantics" nouva-semantics.log env METALPASSTHROUGH_NO_UVA=1 "$MPT" --hugepages 2048 --memory 8192 --runtime "$ROOT/guest/dist" -v "$ROOT/e2e:/e2e" "$RUNTIME" /e2e/cuda/semantics_test

{
    echo "# MetalPassthrough test results"
    echo
    echo "$(date -u '+%Y-%m-%d %H:%M UTC') — $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo
    echo "| Step | Result | Log |"
    echo "|---|---|---|"
    printf "%b\n" "$results" | sed '/^$/d'
    echo
    for f in test1-apple-smi test2-native test3-driver test3-runtime test3-semantics test4-python selftest nouva-driver; do
        echo "## $f"; echo; echo '```'; grep -v '^\[metalpassthrough' "$OUT/$f.log" | tail -40; echo '```'; echo
    done
} > "$OUT/summary.md"

echo "== $pass passed, $fail failed  (summary: $OUT/summary.md)"
[ "$fail" -eq 0 ]
