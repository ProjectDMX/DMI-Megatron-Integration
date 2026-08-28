# Installation

`DMI-Megatron-Integration` must be installed with its pinned `Megatron-LM-DMI` fork and a compatible DMI release. Use a dedicated environment rather than sharing an environment with another DMI framework integration.

## Requirements

The integration requires:

- Python `>=3.12,<3.15`
- DMI `>=1.2.0,<2.0`
- the `third_party/megatron-lm` revision pinned by this repository
- the Transformer Engine source revision declared by the pinned fork in [`third_party/megatron-lm/pyproject.toml`](../third_party/megatron-lm/pyproject.toml) under `[tool.uv.sources]`
- PyTorch, Transformer Engine, the CUDA toolkit, cuDNN, NCCL, and the compiler to use a mutually compatible CUDA stack

Do not replace the pinned fork with an unrelated `megatron-core` release. The integration and fork revisions are versioned as a pair.

## Tested configuration

The following configuration has been tested:

- Python 3.12.14
- PyTorch 2.13.0+cu129
- CUDA toolkit 12.9.1
- cuDNN 9.20.0
- NCCL 2.29.7
- Transformer Engine 2.14.1 at commit `366798ef8a0a00d8f2c1650d11e7e623d7c33e26`

Other compatible dependency combinations may work but have not been tested by this project.

## Clone the complete source tree

Clone recursively so that `third_party/megatron-lm` is checked out at the revision selected by this integration:

```bash
git clone --recurse-submodules https://github.com/ProjectDMX/DMI-Megatron-Integration.git
cd DMI-Megatron-Integration
```

If the repository was cloned without `--recurse-submodules`, initialize the pinned fork before installing:

```bash
git submodule update --init --recursive
```

## Create the Python environment

The following commands create a dedicated Python 3.12 environment and install the tested PyTorch build:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
```

For a different supported dependency combination, select a PyTorch build compatible with that environment's CUDA stack.

## Install Transformer Engine

Transformer Engine is not provided by the Megatron fork's `training` extra. The following commands read the authoritative source URL and revision from the pinned fork, then install its PyTorch integration:

```bash
TE_SOURCE="$(python -c 'import tomllib; source = tomllib.load(open("third_party/megatron-lm/pyproject.toml", "rb"))["tool"]["uv"]["sources"]["transformer-engine"]; print("git+{}@{}".format(source["git"], source["rev"]))')"
NVTE_FRAMEWORK=pytorch python -m pip install --no-build-isolation \
  "$TE_SOURCE"
```

The source build requires Git, CMake, Ninja, a compatible C++ compiler, pybind11, the CUDA toolkit, cuDNN, and NVCC. Refer to Transformer Engine's source installation requirements at the revision selected by the fork when those prerequisites are not already available.

## Install the fork and integration

Install the pinned fork with its training dependencies, then install this integration. Installing the integration resolves its declared DMI requirement, `DMI>=1.2.0,<2.0`:

```bash
python -m pip install -e "third_party/megatron-lm[training]"
python -m pip install -e .
```

Installing Transformer Engine or the Megatron packages does not require rebuilding DMI unless the installation changes the active PyTorch or CUDA ABI.

## Verify the installation

Confirm that DMI, the integration, Megatron Core, PyTorch, and Transformer Engine import from the active environment:

```bash
python - <<'PY'
import dmi
import dmi_megatron_integration
import megatron.core
import torch
import transformer_engine

print("DMI:", dmi.__file__)
print("DMI-Megatron-Integration:", dmi_megatron_integration.__file__)
print("Megatron Core:", megatron.core.__file__)
print("PyTorch:", torch.__version__)
print("Transformer Engine:", transformer_engine.__version__)
PY
```
