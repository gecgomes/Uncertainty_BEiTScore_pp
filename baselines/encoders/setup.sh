#!/usr/bin/env bash
# Code and checkpoints of the encoder-based baselines, installed into the environment of this
# repository (activate it first). Nothing of these baselines is redistributed here: their code is
# cloned from the authors' repositories and their checkpoints downloaded from the authors' links.
#   repos/pacscore  github.com/aimagelab/pacscore, with checkpoints/PAC++_clip_ViT-{B-32,L-14}.pth
#   repos/SPECS     github.com/mbzuai-nlp/SPECS, with checkpoints/spec.pt (Hugging Face Xiaohud/SPECS)
# CLIPScore downloads the LAION CLIP models from Hugging Face on first use; BEiTScore and BEiTScore++
# use the release downloaded by `python scripts/download_release.py models`.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p repos

clone() {  # url folder
    [[ -d "$2/.git" ]] && echo "[keep] $2" || git clone "$1" "$2"
}
download() {  # url file
    if [[ -s "$2" ]]; then echo "[keep] $2"; else
        mkdir -p "$(dirname "$2")"; curl -L --fail --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; fi
}

clone https://github.com/aimagelab/pacscore.git repos/pacscore
# the LoRA CLIP tokenizer looks for the CLIP vocabulary next to it; the repository has it in models/clip/
[[ -s repos/pacscore/models/clip_lora/bpe_simple_vocab_16e6.txt.gz ]] || \
    cp repos/pacscore/models/clip/bpe_simple_vocab_16e6.txt.gz repos/pacscore/models/clip_lora/
download https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-B-32.pth repos/pacscore/checkpoints/PAC++_clip_ViT-B-32.pth
download https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth repos/pacscore/checkpoints/PAC++_clip_ViT-L-14.pth

clone https://github.com/mbzuai-nlp/SPECS.git repos/SPECS
download https://huggingface.co/Xiaohud/SPECS/resolve/main/spec.pt repos/SPECS/checkpoints/spec.pt

# the CLIP tokenizers of PAC-S and SPECS; transformers and huggingface_hub come with this repository
python -m pip install ftfy regex
python -c "import beitscore, ftfy, regex, transformers; print('beitscore found at', beitscore.__file__)"
