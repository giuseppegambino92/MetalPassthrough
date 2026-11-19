#!/bin/bash
# Crash-safety stress test. Runs inside a VM booted with 256 x 2 MiB hugepages (512 MiB pool).
#   crash_stress.sh [rounds] [nokeeper]    (nokeeper: control run, expected to FAIL)
# Each round starts 3 GPU processes (96 MiB arenas, ~2 s kernels writing their whole arena) and
# SIGKILLs them at random moments: during the handshake, during allocation, or with the kernel in
# flight. A probe then waits until the ENTIRE hugepage pool is free again and checks every word of
# it for foreign writes. Then: memory exhaustion (a session that does not fit must be refused
# cleanly, one that fits must work) and a final leak check.
B=$(dirname "$0")/crash_test
ROUNDS=${1:-12}
POOL_MB=$(( $(awk '/HugePages_Total/ {print $2}' /proc/meminfo) * 2 ))
fail=0
rnd() { echo $(( $(od -An -N2 -tu2 /dev/urandom) % $1 )); }
echo "pool: $POOL_MB MiB of hugepages, $ROUNDS rounds"
for r in $(seq 1 "$ROUNDS"); do
    pids=""
    for v in 1 2 3; do
        if [ "$2" = nokeeper ]; then METALPASSTHROUGH_NO_KEEPER=1 $B victim 96 > /tmp/v$v.out 2>&1 &
        else $B victim 96 > /tmp/v$v.out 2>&1 & fi
        pids="$pids $!"
    done
    # Every third round kills within 80 ms (handshake / arena mapping / allocation); the others
    # anywhere in 0-2.5 s (compile, launch, kernel in flight).
    if [ $((r % 3)) -eq 1 ]; then ms=$(rnd 80); else ms=$(rnd 2500); fi
    sleep "$(printf '0.%03d' $((ms % 1000)))"; [ "$ms" -ge 1000 ] && sleep $((ms / 1000))
    ready=$(cat /tmp/v*.out | grep -c READY)
    kill -9 $pids 2>/dev/null; wait $pids 2>/dev/null
    out=$($B probe $((POOL_MB - 8)) 400 full)
    rc=$?
    echo "round $r: killed after ${ms} ms ($ready/3 kernels in flight) | $out"
    [ $rc -ne 0 ] && fail=1
done

[ "$2" = nokeeper ] && { [ $fail -eq 0 ] && echo "crash_stress (nokeeper control): no corruption seen" || echo "crash_stress (nokeeper control): corruption detected, as expected without the keeper"; exit 0; }
echo "-- memory exhaustion"
$B hog $((POOL_MB - 64)) > /tmp/hog.out & H=$!
for i in $(seq 1 100); do grep -q HOG /tmp/hog.out && break; sleep 0.05; done
$B open 128; rc=$?; [ $rc -eq 3 ] || { echo "FAIL: oversized session was not refused cleanly (rc $rc)"; fail=1; }
$B open 48; rc=$?; [ $rc -eq 0 ] || { echo "FAIL: a fitting session did not work under pressure (rc $rc)"; fail=1; }
$B victim 48 > /tmp/v.out 2>&1 & V=$!
for i in $(seq 1 100); do grep -q READY /tmp/v.out && break; sleep 0.05; done
kill -9 $V; wait $V 2>/dev/null
kill -9 $H; wait $H 2>/dev/null
out=$($B probe $((POOL_MB - 8)) 400 full) || fail=1
echo "after exhaustion + kill: $out"

echo "-- leaks"
sleep 0.5
free=$(awk '/HugePages_Free/ {print $2}' /proc/meminfo); total=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)
echo "hugepages free after all rounds: $free / $total"
[ "$free" -eq "$total" ] || { echo "FAIL: $((total - free)) hugepages leaked"; fail=1; }
$B open 256 || fail=1
[ $fail -eq 0 ] && echo "crash_stress: PASS" || echo "crash_stress: FAILED"
exit $fail
