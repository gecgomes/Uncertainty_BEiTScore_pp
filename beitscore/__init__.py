"""BEiTScore and BEiTScore++: reference-free image captioning evaluation with BEiT-3.

- BEiTScore (arXiv preprint): a BEiT-3 cross-encoder that predicts one quality score.
- BEiTScore++ (Findings of EMNLP 2026): the same model with a quantile head that
  also predicts a calibrated confidence interval.
"""

__version__ = "0.1.0.dev0"

__all__ = ["BEiTScore", "BEiTScorePP"]


def __getattr__(name):
    # Imported lazily, so that `import beitscore` stays light.
    if name in __all__:
        from . import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module 'beitscore' has no attribute {name!r}")
