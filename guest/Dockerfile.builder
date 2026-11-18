# MetalPassthrough guest builder: Linux arm64 toolchain + NVIDIA CUDA 12.9 headers/nvcc (used only to
# compile against the official CUDA ABI headers and to produce real PTX / CUDA test binaries).
FROM docker.io/library/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      build-essential wget ca-certificates python3 python3-numpy cargo rustc pkg-config file && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
WORKDIR /src
