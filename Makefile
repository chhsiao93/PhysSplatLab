.PHONY: install install-base

# Full install: base deps + CUDA extensions compiled for sm_120 (RTX 5090)
install: install-base
	TORCH_CUDA_ARCH_LIST="12.0" uv pip install --no-build-isolation \
		gaussian-splatting/submodules/diff-gaussian-rasterization \
		gaussian-splatting/submodules/simple-knn \
		gaussian-splatting/submodules/fused-ssim

# Base install: torch, warp-lang, and all other deps
install-base:
	uv sync
