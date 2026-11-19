#!/bin/sh
# Runs inside the VM (booted with 160 x 2 MiB hugepages). The victim takes 300 MiB, so the probe's
# 280 MiB can only be satisfied by the victim's own pages once they are released.
# Usage: crash_test.sh [nokeeper]
B=$(dirname "$0")/crash_test
if [ "$1" = nokeeper ]; then export METALPASSTHROUGH_NO_KEEPER=1; fi
$B victim 300 > /tmp/victim.out 2>&1 &
V=$!
for i in $(seq 1 300); do grep -q READY /tmp/victim.out && break; sleep 0.05; done
cat /tmp/victim.out
kill -9 $V
unset METALPASSTHROUGH_NO_KEEPER
$B probe 280 1500
