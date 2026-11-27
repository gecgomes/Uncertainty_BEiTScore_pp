#!/usr/bin/env bash
# Environment of FLEUR, as its README asks (the LLaVA install instructions): Python 3.10 and the LLaVA
# package of the commit FLEUR pins, LLaVA 1.1.3 (torch 2.0.1, transformers 4.31.0), installed with pip install -e.
#   PYTHON_BIN   Python 3.10 interpreter (default: python3.10)
#   TORCH_INDEX  wheel index for torch if the default CUDA build does not suit your GPU,
#                e.g. https://download.pytorch.org/whl/cu118
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN=${PYTHON_BIN:-python3.10}

if [[ -d FLEUR/.git ]]; then
    git -C FLEUR submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/Yebin46/FLEUR.git FLEUR
fi
[[ -d FLEUR/LLaVA/llava ]] || { echo "the LLaVA submodule of FLEUR was not fetched"; exit 1; }
echo "LLaVA commit: $(git -C FLEUR/LLaVA rev-parse --short HEAD)"

[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX:-}" ]]; then
    python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url "$TORCH_INDEX"
fi
python -m pip install -e FLEUR/LLaVA
python -m pip install "numpy<2" "protobuf<5" scipy pandas
python -c "import llava, torch, transformers; print('llava', torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available())"
