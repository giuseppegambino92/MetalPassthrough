#!/bin/sh
# Builds and ad-hoc signs the host binaries (Virtualization.framework requires the entitlement).
set -e
cd "$(dirname "$0")/.."
CONFIG=${1:-release}
swift build -c "$CONFIG"
BIN=$(swift build -c "$CONFIG" --show-bin-path)
codesign --force --sign - --timestamp=none --entitlements=signing/mpt.entitlements "$BIN/mpt-run"
echo "$BIN/mpt-run"
