.PHONY: install install-base patch-submodules

# Full install: base deps + CUDA extensions.
# Default targets sm_120 (RTX 5090); override for other GPUs, e.g.
#   make install TORCH_CUDA_ARCH_LIST=9.0   # GH200 / Hopper (TACC Vista)
TORCH_CUDA_ARCH_LIST ?= 12.0

install: install-base patch-submodules
	TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" uv pip install --no-build-isolation \
		gaussian-splatting/submodules/diff-gaussian-rasterization \
		gaussian-splatting/submodules/simple-knn \
		gaussian-splatting/submodules/fused-ssim

# Base install: torch, warp-lang, and all other deps
install-base:
	uv sync
	# Safety net: on some uv versions, `uv sync` can silently skip installing
	# the project itself on a first-ever install (seen on uv 0.11.5 on TACC
	# Vista). This is a cheap no-op when uv sync already got it right.
	uv pip install --no-deps -e .

# Patch upstream submodule bugs we can't push upstream (see patches/)
patch-submodules:
	./scripts/apply-patches.sh
