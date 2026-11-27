#!/usr/bin/env bash
# Environment of EXPERT, as its README asks (the LLaVA install instructions): Python 3.10 and the LLaVA
# package of the commit EXPERT pins, LLaVA 1.2.2 (torch 2.1.2, transformers 4.37.2), installed with pip install -e.
#   PYTHON_BIN   Python 3.10 interpreter (default: python3.10)
#   TORCH_INDEX  wheel index for torch if the default CUDA build does not suit your GPU,
#                e.g. https://download.pytorch.org/whl/cu118
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN=${PYTHON_BIN:-python3.10}

if [[ -d EXPERT/.git ]]; then
    git -C EXPERT submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/hjkim811/EXPERT.git EXPERT
fi
[[ -d EXPERT/LLaVA/llava ]] || { echo "the LLaVA submodule of EXPERT was not fetched"; exit 1; }
echo "LLaVA commit: $(git -C EXPERT/LLaVA rev-parse --short HEAD)"

[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX:-}" ]]; then
    python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_INDEX"
fi
python -m pip install -e EXPERT/LLaVA
python -m pip install "numpy<2" "protobuf<5" "peft==0.12.0" scipy pandas
python -c "import llava, torch, transformers; print('llava', torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available())"
