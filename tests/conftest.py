import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_cfg():
    """A BEiT-3 config small enough to run on CPU in a few seconds."""
    from torchscale.architecture.config import EncoderConfig

    return EncoderConfig(
        img_size=32, patch_size=16, vocab_size=100, multiway=True,
        layernorm_embedding=False, normalize_output=False, no_output_layer=True,
        drop_path_rate=0.0, encoder_embed_dim=64, encoder_attention_heads=2,
        encoder_ffn_embed_dim=128, encoder_layers=2,
    )


@pytest.fixture
def tiny_batch():
    torch.manual_seed(0)
    return dict(
        image=torch.randn(4, 3, 32, 32),
        text_description=torch.randint(3, 100, (4, 7)),
        padding_mask=torch.zeros(4, 7, dtype=torch.long),
    )


def load_script(rel_path, name):
    """Import a script as a module without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
