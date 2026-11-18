# MetalPassthrough — zero-copy Metal / CUDA-PTX GPU passthrough for apple/container Linux VMs.
# Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

CONFIG    ?= release
BUILDER   ?= metalpassthrough-builder:latest
RUNTIME   ?= metalpassthrough-runtime:latest
GUEST_DIR := $(HOME)/Library/Application Support/dev.metalpassthrough/guest
PREFIX    ?= $(HOME)/.local
CTR       := container run --rm -v $(CURDIR):/src

.PHONY: all host guest images e2e ptx test test-all install uninstall patch clean help

all: host guest e2e

help:
	@echo "make images      build the Linux builder/runtime images with apple/container"
	@echo "make host        build + ad-hoc sign mpt-run (Virtualization entitlement)"
	@echo "make guest       build libmetalpassthrough, libcuda.so.1, libcudart.so.12, apple-smi, python pkg"
	@echo "make e2e         build the end-to-end test programs (CUDA driver/runtime, native Metal)"
	@echo "make test        host unit tests (PTX->MSL translator on the local GPU)"
	@echo "make test-all    unit tests + the 4 end-to-end tests inside real Linux containers"
	@echo "make install     install mpt-run to $(PREFIX)/bin and the guest runtime used by --gpu"
	@echo "make patch       apply patches/ to vendor/container (apple/container 1.7.1) and build it"

host:
	./scripts/build-host.sh $(CONFIG)

images:
	cd guest && container build -t $(BUILDER) -f Dockerfile.builder .
	cd guest && container build -t $(RUNTIME) -f Dockerfile.runtime .

guest:
	$(CTR) -w /src/guest $(BUILDER) make -B all   # -B: VirtioFS mtimes vs guest clock skew

e2e: guest
	$(CTR) -w /src/e2e $(BUILDER) sh -c '\
	  gcc -O2 -o cuda/driver_test cuda/driver_test.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -lm && \
	  nvcc -O3 -cudart shared -o cuda/runtime_test cuda/runtime_test.cu && \
	  nvcc -O3 -cudart shared -o cuda/semantics_test cuda/semantics_test.cu -lcuda && \
	  gcc -O2 -o native/native_test native/native_test.c -I/src/guest/dist/include -L/src/guest/dist/lib -lmetalpassthrough -lm'

# Regenerates the nvcc PTX fixture (CUDA 13.5 toolkit, Debian 12 arm64).
ptx:
	cd e2e/cuda && container run --rm -v $$PWD:/w -w /w docker.io/library/debian:12 sh build-ptx.sh
	cp e2e/cuda/kernels.ptx Tests/MetalPassthroughHostTests/Fixtures/kernels.ptx

test:
	swift test

test-all:
	./scripts/test-all.sh

install: host guest
	mkdir -p "$(PREFIX)/bin" "$(GUEST_DIR)"
	install -m 755 "$$(swift build -c $(CONFIG) --show-bin-path)/mpt-run" "$(PREFIX)/bin/mpt-run"
	codesign --force --sign - --timestamp=none --entitlements=signing/mpt.entitlements "$(PREFIX)/bin/mpt-run"
	rsync -a --delete guest/dist/ "$(GUEST_DIR)/"
	@echo "installed $(PREFIX)/bin/mpt-run and the guest runtime in $(GUEST_DIR)"

uninstall:
	rm -f "$(PREFIX)/bin/mpt-run"
	rm -r "$(GUEST_DIR)"

# Builds apple/container with the --gpu patch (MetalPassthrough resolved from this checkout).
patch:
	test -d vendor/container || git clone -q --depth 1 -b 1.7.1 https://github.com/apple/container.git vendor/container
	cd vendor/container && (git apply --check ../../patches/0001-feat-gpu-passthrough-metal-zero-copy.patch 2>/dev/null && \
	  git apply ../../patches/0001-feat-gpu-passthrough-metal-zero-copy.patch || echo "patch already applied")
	cd vendor/container && METALPASSTHROUGH_PATH=$(CURDIR) $(MAKE) container

clean:
	swift package clean
	rm -r guest/dist guest/apple-smi/target build 2>/dev/null || true
	rm -f e2e/cuda/driver_test e2e/cuda/runtime_test e2e/cuda/semantics_test e2e/native/native_test
