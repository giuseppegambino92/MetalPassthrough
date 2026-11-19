# MetalPassthrough guest runtime (Linux aarch64). Built natively inside the metalpassthrough-builder image:
#   make -C .. guest        (from the repo root, on the Mac)
# Output: dist/{lib,bin,include,python}, bind-mounted into containers at /usr/local/metalpassthrough.
CC      ?= gcc
CUDA    ?= /usr/local/cuda
CFLAGS  ?= -O2 -g
CFLAGS  += -fPIC -std=gnu11 -Wall -Wextra -Wno-unused-parameter -Iinclude -I../include -fvisibility=hidden -D_GNU_SOURCE
DIST    := dist

LIBS := $(DIST)/lib/libmetalpassthrough.so.1 $(DIST)/lib/libcuda.so.1 $(DIST)/lib/libcudart.so.12 $(DIST)/lib/libcublas.so.12
BINS := $(DIST)/bin/mpt-selftest $(DIST)/bin/apple-smi

all: $(LIBS) $(BINS) headers python

$(DIST)/lib/libmetalpassthrough.so.1: src/mpt.c include/metalpassthrough.h ../include/metalpassthrough_protocol.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -Wl,-soname,libmetalpassthrough.so.1 -o $@ src/mpt.c ../Sources/MPTProtocol/abi_check.c -lpthread
	ln -sf libmetalpassthrough.so.1 $(DIST)/lib/libmetalpassthrough.so

# Drop-in CUDA Driver API, compiled against NVIDIA's official cuda.h for exact ABI compatibility.
$(DIST)/lib/libcuda.so.1: src/cuda_driver.c src/fatbin.c src/fatbin.h $(DIST)/lib/libmetalpassthrough.so.1
	$(CC) $(CFLAGS) -I$(CUDA)/include -shared -Wl,-soname,libcuda.so.1 -o $@ src/cuda_driver.c src/fatbin.c \
	    -L$(DIST)/lib -lmetalpassthrough -Wl,-rpath,'$$ORIGIN' -lpthread -ldl
	ln -sf libcuda.so.1 $(DIST)/lib/libcuda.so

# CUDA Runtime API (incl. nvcc fatbinary registration / <<<>>> launches).
$(DIST)/lib/libcudart.so.12: src/cuda_runtime.c src/fatbin.h src/libcudart.map $(DIST)/lib/libcuda.so.1
	$(CC) $(CFLAGS) -I$(CUDA)/include -shared -Wl,-soname,libcudart.so.12 -Wl,--version-script=src/libcudart.map -o $@ src/cuda_runtime.c \
	    -L$(DIST)/lib -lcuda -lmetalpassthrough -Wl,-rpath,'$$ORIGIN' -lpthread
	ln -sf libcudart.so.12 $(DIST)/lib/libcudart.so

# cuBLAS subset (SGEMM via host MPS, level-1 via Metal kernels), against the official <cublas_v2.h>.
$(DIST)/lib/libcublas.so.12: src/cublas.c src/libcublas.map $(DIST)/lib/libcuda.so.1
	$(CC) $(CFLAGS) -I$(CUDA)/include -shared -Wl,-soname,libcublas.so.12 -Wl,--version-script=src/libcublas.map -o $@ src/cublas.c \
	    -L$(DIST)/lib -lcuda -lmetalpassthrough -Wl,-rpath,'$$ORIGIN' -lm
	ln -sf libcublas.so.12 $(DIST)/lib/libcublas.so

$(DIST)/bin/mpt-selftest: tests/selftest.c $(DIST)/lib/libmetalpassthrough.so.1
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< -L$(DIST)/lib -lmetalpassthrough -Wl,-rpath,/usr/local/metalpassthrough/lib -lm

$(DIST)/bin/apple-smi: apple-smi/src/main.rs apple-smi/Cargo.toml $(DIST)/lib/libmetalpassthrough.so.1
	@mkdir -p $(@D)
	cd apple-smi && METALPASSTHROUGH_LIB_DIR=$(abspath $(DIST)/lib) cargo build --release --offline -q
	cp apple-smi/target/release/apple-smi $@

headers:
	@mkdir -p $(DIST)/include
	cp include/metalpassthrough.h ../include/metalpassthrough_protocol.h $(DIST)/include/

python:
	@mkdir -p $(DIST)/python
	rm -rf $(DIST)/python/metalpassthrough && cp -r python/metalpassthrough $(DIST)/python/

clean:
	rm -rf $(DIST) apple-smi/target

.PHONY: all headers python clean
