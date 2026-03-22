# PhysSplatLab

## Requirements

- Linux
- CUDA 12.8
- NVIDIA GPU with sm_120 architecture (RTX 5090)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

**1. Clone the repository with submodules**

```bash
git clone --recurse-submodules <repo-url>
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

This will:
- Create a virtual environment at `.venv`
- Install PyTorch (CUDA 12.8), warp-lang, and all other dependencies
- Compile and install the gaussian-splatting CUDA extensions against the installed torch

**3. Activate the environment**

```bash
source .venv/bin/activate
```
