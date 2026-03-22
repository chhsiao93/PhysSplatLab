#!/bin/bash
set -e

# Base install: torch, warp-lang, and all other deps
uv sync

# Build and install CUDA extensions for sm_120 (RTX 5090)
TORCH_CUDA_ARCH_LIST="12.0" uv pip install --no-build-isolation \
    gaussian-splatting/submodules/diff-gaussian-rasterization \
    gaussian-splatting/submodules/simple-knn \
    gaussian-splatting/submodules/fused-ssim
