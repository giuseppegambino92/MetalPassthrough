#!/bin/sh
# Generates real NVIDIA PTX (and an nvcc-built CUDA Runtime test binary) from the .cu sources using
# NVIDIA's official CUDA toolkit for Linux arm64 (sbsa), inside a Linux container on the Mac:
#   container run --rm -v $PWD:/w -w /w docker.io/library/debian:12 sh build-ptx.sh
set -e
export DEBIAN_FRONTEND=noninteractive
if ! command -v /usr/local/cuda/bin/nvcc >/dev/null; then
  apt-get update -qq && apt-get install -y -qq wget ca-certificates g++ >/dev/null
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/sbsa/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && rm cuda-keyring_1.1-1_all.deb
  apt-get update -qq && apt-get install -y -qq cuda-nvcc-13-5 cuda-cudart-dev-13-5 >/dev/null
fi
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -2
nvcc -ptx -O3 -arch=compute_80 -o kernels.ptx kernels.cu
echo "wrote kernels.ptx ($(wc -l < kernels.ptx) lines)"
