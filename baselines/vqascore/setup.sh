#!/usr/bin/env bash
# Environment of VQAScore: the t2v_metrics package (github.com/linzhiqiu/t2v_metrics), which downloads
# CLIP-FlanT5-XXL from Hugging Face on first use.
# Pins (t2v_metrics imports every model family on import, so their dependencies must install too):
#   t2v_metrics 1.2     3.0 depends on "apex==0.9.10dev", an unrelated PyPI package that fails to build
#   torch 2.1.2         as in the t2v_metrics README
#   image-reward 1.5    otherwise pip falls back to the 1.0 sdist, which needs pkg_resources and fails
#   openai CLIP         imported by image-reward, not on PyPI
#   diffusers 0.30.3    newer versions call torch.xpu, missing in torch 2.1.2
#   numpy < 2           torch 2.1.2 was built against numpy 1
#   PYTHON_BIN   Python 3.10 interpreter (default: python3.10)
#   TORCH_INDEX  optional PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu118
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN=${PYTHON_BIN:-python3.10}
[[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
if [[ -n "${TORCH_INDEX:-}" ]]; then
    python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_INDEX"
fi
PINS=("torch==2.1.2" "torchvision==0.16.2" "numpy<2" "image-reward==1.5" "diffusers==0.30.3")
python -m pip install "t2v_metrics==1.2" "${PINS[@]}" scipy scikit-learn pandas
python -m pip install --no-build-isolation "git+https://github.com/openai/CLIP.git" "${PINS[@]}"
python -c "import t2v_metrics, torch; assert 'clip-flant5-xxl' in t2v_metrics.list_all_vqascore_models(); print('t2v_metrics ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
