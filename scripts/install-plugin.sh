#!/bin/sh
# Installs MetalPassthrough as an apple/container *runtime plugin* next to an unmodified `container`
# (e.g. Homebrew's). No CLI patch: containers opt in with the stock flag
#     container run --runtime container-runtime-metalpassthrough <image> ...
# The plugin is apple/container's own Linux runtime helper, built from the same release tag as the
# installed CLI, with the MetalPassthrough attachment compiled in.
#   scripts/install-plugin.sh            build + install + restart the container services
#   scripts/install-plugin.sh --uninstall
# Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
set -eu
cd "$(dirname "$0")/.."
ROOT=$PWD
NAME=container-runtime-metalpassthrough
CLI=$(command -v container) || { echo "apple/container is not installed (brew install container)" >&2; exit 1; }
INSTALL_ROOT=$(cd "$(dirname "$(readlink -f "$CLI")")/.." && pwd)
PLUGINS="$INSTALL_ROOT/libexec/container-plugins"
# Homebrew links this to $(brew --prefix)/libexec/container-plugins (kept across upgrades); the
# loader only scans it when the target exists.
if [ -L "$PLUGINS" ]; then
    T=$(readlink "$PLUGINS")
    case "$T" in /*) ;; *) T="$(dirname "$PLUGINS")/$T" ;; esac
    mkdir -p "$T"
fi
DEST="$PLUGINS/$NAME"

if [ "${1:-}" = "--uninstall" ]; then
    rm -rf "$DEST"
    container system stop >/dev/null 2>&1 || true
    container system start >/dev/null
    echo "removed $DEST"
    exit 0
fi

VERSION=$(container --version | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p')
echo "container CLI $VERSION at $INSTALL_ROOT"
if [ ! -d vendor/container ]; then
    git clone -q --depth 1 -b "$VERSION" https://github.com/apple/container.git vendor/container
fi
TAG=$(git -C vendor/container describe --tags --abbrev=0)
[ "$TAG" = "$VERSION" ] || echo "warning: vendor/container is $TAG but the installed CLI is $VERSION (XPC types must match)" >&2
if ! grep -q MetalPassthroughAttachment vendor/container/Sources/Services/RuntimeLinux/Server/RuntimeService.swift; then
    (cd vendor/container && git apply ../../patches/0001-feat-gpu-passthrough-metal-zero-copy.patch \
        && git apply ../../patches/0002-runtime-plugin.patch)
fi

make guest >/dev/null
# Per-session GPU worker (process isolation): built from this package, installed next to the helper.
swift build -c release --product mpt-gpu-worker
WORKER="$(swift build -c release --show-bin-path)/mpt-gpu-worker"
(cd vendor/container && METALPASSTHROUGH_PATH="$ROOT" swift build -c release --product container-runtime-linux)
BIN="$(cd vendor/container && swift build -c release --show-bin-path)/container-runtime-linux"

rm -rf "$DEST"
mkdir -p "$DEST/bin" "$DEST/share/metalpassthrough"
cp "$BIN" "$DEST/bin/$NAME"
codesign --force --sign - --entitlements signing/mpt.entitlements "$DEST/bin/$NAME"
cp "$WORKER" "$DEST/bin/mpt-gpu-worker"
codesign --force --sign - "$DEST/bin/mpt-gpu-worker"   # needs no entitlement: Metal only
cp -R guest/dist "$DEST/share/metalpassthrough/guest"
cat > "$DEST/config.toml" <<TOML
abstract = "Linux container runtime with MetalPassthrough zero-copy Apple GPU access"
author = "MetalPassthrough"
version = 0.1

[servicesConfig]
loadAtBoot = false
runAtLoad = false
defaultArguments = []

[[servicesConfig.services]]
type = "runtime"
TOML

echo "installed $DEST; restarting container services (running containers are stopped)"
container system stop >/dev/null 2>&1 || true
container system start >/dev/null
echo "try:  container run --rm --runtime $NAME metalpassthrough-runtime:latest apple-smi"
