# PhysSplatLab

## Requirements

- Linux (x86_64 or aarch64)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A CUDA toolkit (`nvcc`) matching your platform, needed to build the
  gaussian-splatting extensions and for warp-lang's runtime kernel compilation:

  | Platform | GPU | CUDA | PyTorch index |
  |---|---|---|---|
  | x86_64 (e.g. Lambda) | RTX 5090 (sm_120, Blackwell) | 12.8 | `cu128` |
  | aarch64 (e.g. TACC Vista) | GH200 (sm_90, Hopper) | 12.6 | `cu126` |

  `pyproject.toml` picks the right PyTorch index automatically based on
  `platform_machine`, so no manual edits are needed there. On Vista, make
  sure `nvcc` and a compatible host compiler are on `PATH` before installing
  and before running anything (warp-lang JIT-compiles kernels at runtime
  too):

  ```bash
  module load cuda/12.6
  module load gcc/13.2.0   # NOT the default gcc (15.1.0) or the nvidia/nvc++
                            # toolchain that's loaded by default on the GH
                            # nodes — CUDA 12.6's nvcc only supports GCC up
                            # to 13.x as a host compiler
  which nvcc gcc g++       # sanity check before installing
  ```

  If `CC`/`CXX` are already exported to something else in your shell, unset
  or override them after loading the modules above:
  ```bash
  export CC=$(which gcc) CXX=$(which g++)
  ```

## Setup

**1. Clone the repository with submodules**

```bash
git clone --recurse-submodules git@github.com:chhsiao93/PhysSplatLab.git
cd PhysSplatLab
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

**2. Install the environment**

```bash
./install.sh
```

Or with `make`:

```bash
make install
```

This defaults to compiling the CUDA extensions for sm_120 (RTX 5090). On
other GPUs, override `TORCH_CUDA_ARCH_LIST`, e.g. on TACC Vista (GH200,
sm_90):

```bash
TORCH_CUDA_ARCH_LIST=9.0 ./install.sh
# or
make install TORCH_CUDA_ARCH_LIST=9.0
```

This will:
- Create a virtual environment at `.venv`
- Install PyTorch (matching your platform's CUDA index), warp-lang, and all
  other dependencies
- Compile and install the gaussian-splatting CUDA extensions against the
  installed torch

**3. Activate the environment**

```bash
source .venv/bin/activate
```

**Run scripts with plain `python`, not `uv run`.** The gaussian-splatting
CUDA extensions (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`)
are intentionally not listed in `pyproject.toml`'s dependencies (see the
NOTE there) since they need `torch` already installed before they can be
built. Because of that, `uv run` — which auto-syncs the environment against
`pyproject.toml` first — will uninstall them before running your script. Use:

```bash
python examples/01_load_and_render.py
```

not `uv run examples/01_load_and_render.py`. If you do need `uv run`
(e.g. to also pick up a newly added dependency), pass `--no-sync`, or just
rerun `./install.sh` / `make install` afterwards to rebuild the extensions.

## Examples

See [examples/](examples/) for runnable scripts covering the core API:
loading and rendering a splat, orbiting a camera around a scene into a video,
and running an MPM physics simulation with per-region material assignment.
